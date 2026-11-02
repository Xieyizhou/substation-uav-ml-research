"""Publication boundaries are identical for ZIP exports and clean copies."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.export_source import export_source
from src.ml.artifacts import git_commit, object_sha256
from src.sandbox.beta_install import _copy_source
from src.sandbox.source_bundle import MANIFEST, PROVENANCE, source_paths


class SourceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "README.md").write_text("source\n")
        (self.root / "examples/demo").mkdir(parents=True)
        (self.root / "examples/demo/input.json").write_text('{"demo":true}\n')
        (self.root / "requirements-offline.txt").write_text("Pillow>=10\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)

    def unpack(self):
        output = Path(self.temporary.name) / "source.zip"
        report = export_source(self.root, output)
        target = self.root / "unpacked"
        with zipfile.ZipFile(output) as archive:
            archive.extractall(target)
        return report, target

    def test_default_inventory_excludes_untracked_and_includes_public_examples(self):
        (self.root / "scripts").mkdir()
        (self.root / "scripts/local_only.py").write_text("LOCAL = True\n")
        names = [name for name, _ in source_paths(self.root)]
        self.assertIn("examples/demo/input.json", names)
        self.assertIn("requirements-offline.txt", names)
        self.assertNotIn("scripts/local_only.py", names)
        self.assertIn("scripts/local_only.py", [
            name for name, _ in source_paths(self.root, include_untracked=True)
        ])

    def test_export_and_install_copy_have_identical_manifests(self):
        report, unpacked = self.unpack()
        target = Path(self.temporary.name) / "install"
        target.mkdir()
        copied = _copy_source(self.root, target)
        self.assertEqual(report["source_manifest_sha256"], copied["source_manifest_sha256"])
        self.assertEqual((unpacked / MANIFEST).read_bytes(), (target / MANIFEST).read_bytes())
        self.assertEqual((unpacked / PROVENANCE).read_bytes(), (target / PROVENANCE).read_bytes())
        again = Path(self.temporary.name) / "from-zip"
        again.mkdir()
        copied_again = _copy_source(unpacked, again)
        self.assertEqual(copied, copied_again)

    def test_unpacked_source_and_empty_child_never_borrow_parent_git_identity(self):
        subprocess.run([
            "git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        ], cwd=self.root, check=True)
        revision = git_commit(self.root)
        self.assertEqual(len(revision), 40)
        _, unpacked = self.unpack()
        self.assertEqual(git_commit(unpacked), revision)
        empty = self.root / "empty-child"
        empty.mkdir()
        self.assertEqual(git_commit(empty), "unknown")

    def test_unpacked_bundle_refuses_changed_source(self):
        subprocess.run([
            "git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        ], cwd=self.root, check=True)
        _, unpacked = self.unpack()
        self.assertEqual(git_commit(unpacked), git_commit(self.root))
        (unpacked / "README.md").write_text("modified")
        with self.assertRaisesRegex(ValueError, "content changed"):
            source_paths(unpacked)
        self.assertEqual(git_commit(unpacked), "unknown")

    def test_manifest_cannot_escape_the_bundle_even_with_recomputed_identity(self):
        _, unpacked = self.unpack()
        rows = json.loads((unpacked / MANIFEST).read_text())
        rows[0]["path"] = "../README.md"
        provenance = json.loads((unpacked / PROVENANCE).read_text())
        provenance["source_manifest_sha256"] = object_sha256(rows)
        (unpacked / MANIFEST).write_text(json.dumps(rows))
        (unpacked / PROVENANCE).write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            source_paths(unpacked)

    def test_export_detects_source_change_between_inventory_and_write(self):
        from scripts import export_source as module
        original = module.describe_source

        def change(files, revision):
            result = original(files, revision)
            (self.root / "README.md").write_text("changed during export")
            return result

        with patch.object(module, "describe_source", side_effect=change):
            with self.assertRaisesRegex(ValueError, "while exporting"):
                export_source(self.root, Path(self.temporary.name) / "changed.zip")
