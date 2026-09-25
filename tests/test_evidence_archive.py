import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from src.ml.artifacts import file_sha256, object_sha256
from src.sandbox.evidence_archive import EvidenceArchive, freeze_evidence


class EvidenceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original/core.py"
        self.source.parent.mkdir()
        self.source.write_text("original implementation\n")
        self.receipt = self.source.parent / "receipt.json"
        self.receipt.write_text(json.dumps({"status": "verified", "inputs": {
            str(self.source): file_sha256(self.source)}}))
        self.output = self.root / "snapshot.zip"

    def freeze(self):
        return freeze_evidence({"flight": self.receipt}, self.output, project_root=self.root)

    def rewrite(self, transform):
        with zipfile.ZipFile(self.output) as archive:
            items = {name: archive.read(name) for name in archive.namelist()}
        transform(items)
        with zipfile.ZipFile(self.output, "w") as archive:
            for name, payload in items.items():
                archive.writestr(name, payload)

    def test_move_and_delete_originals_preserves_exact_receipt_and_source(self):
        original = self.receipt.read_bytes()
        info = self.freeze()
        shutil.rmtree(self.source.parent)
        moved = self.root / "moved.zip"
        self.output.rename(moved)
        with EvidenceArchive(moved, expected_identity=info["manifest_sha256"]) as archive:
            self.assertEqual(archive.verify_all()["records"], 1)
            self.assertEqual(archive.read(str(self.receipt)), original)
            self.assertEqual(archive.read(str(self.source)), b"original implementation\n")
            self.assertEqual(archive.record("flight")["status"], "verified")

    def test_stale_source_cannot_be_frozen_and_no_partial_archive_remains(self):
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_existing_archive_is_not_overwritten(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(self.output.read_bytes(), b"keep")

    def test_changed_object_rejected_even_when_zip_checksum_is_valid(self):
        info = self.freeze()
        key = "objects/" + file_sha256(self.source)
        self.rewrite(lambda items: items.__setitem__(key, b"x" * len(items[key])))
        with EvidenceArchive(self.output, expected_identity=info["manifest_sha256"]) as archive:
            with self.assertRaisesRegex(ValueError, "object changed"):
                archive.verify_all()

    def test_recomputed_manifest_requires_a_new_external_identity(self):
        info = self.freeze()
        def change(items):
            manifest = json.loads(items["manifest.json"])
            manifest["records"]["other"] = manifest["records"].pop("flight")
            items["manifest.json"] = json.dumps(manifest).encode()
        self.rewrite(change)
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            EvidenceArchive(self.output, expected_identity=info["manifest_sha256"])

    def test_missing_object_and_path_escape_members_are_rejected(self):
        info = self.freeze()
        self.rewrite(lambda items: items.__setitem__("../escaped.py", b"bad"))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            EvidenceArchive(self.output, expected_identity=info["manifest_sha256"])
        self.rewrite(lambda items: (items.pop("../escaped.py"), items.pop("objects/" + file_sha256(self.source))))
        with self.assertRaisesRegex(ValueError, "Missing"):
            EvidenceArchive(self.output, expected_identity=info["manifest_sha256"])

    def test_budget_and_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            freeze_evidence({"flight": self.receipt}, self.output, project_root=self.root, max_bytes=1)
        target = self.root / "target"
        self.source.rename(target)
        self.source.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.freeze()

    def test_undeclared_dependency_cannot_be_accepted_by_manifest_only(self):
        info = self.freeze()
        identity = []
        def change(items):
            manifest = json.loads(items["manifest.json"])
            row = manifest["files"].pop(str(self.source))
            items.pop("objects/" + row["sha256"])
            identity.append(object_sha256(manifest))
            items["manifest.json"] = json.dumps(manifest).encode()
        self.rewrite(change)
        with EvidenceArchive(self.output, expected_identity=identity[0]) as archive:
            with self.assertRaisesRegex(ValueError, "dependency missing"):
                archive.verify_all()


if __name__ == "__main__":
    unittest.main()
