"""Lossless archival must preserve evidence and reject incomplete sources."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.camera_archive import (
    ARCHIVE, archive_camera_recording, inspect_camera_archive,
    restore_camera_recording,
)


class CameraArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.paths = []
        for color in (b"abc", b"xyz"):
            payload = color * 10000
            digest = hashlib.sha256(payload).hexdigest()
            path = self.root / f"payloads/frames/{digest}.raw"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.paths.append(path)
        self.record = {"status": "live_complete_not_flight_certified",
                       "training_admitted": False, "control_authority": "none",
                       "inputs": {str(p): file_sha256(p) for p in self.paths}}
        self.save_completion()
        (self.root / "best.pt").write_bytes(b"old model must remain")

    def save_completion(self):
        write_json(self.root / "completion.json",
                   {**self.record, "identity": object_sha256(self.record)})

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_compaction_and_idempotent_restore(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.paths}
        completion = (self.root / "completion.json").read_bytes()
        result = archive_camera_recording(self.root, compact=True)
        self.assertLess(result["archive_bytes"], result["raw_bytes"])
        self.assertTrue(all(not p.exists() for p in self.paths))
        inspect_camera_archive(self.root)
        self.assertEqual(restore_camera_recording(self.root)["restored_count"], 2)
        self.assertEqual(restore_camera_recording(self.root)["restored_count"], 0)
        for path, (content, modified) in before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, modified)
        self.assertEqual((self.root / "completion.json").read_bytes(), completion)
        self.assertEqual((self.root / "best.pt").read_bytes(), b"old model must remain")
        archive_camera_recording(self.root, compact=True)
        self.assertEqual(archive_camera_recording(self.root, compact=True)["removed_loose_bytes"], 0)

    def test_default_only_creates_verified_archive(self):
        archive_camera_recording(self.root)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_rejects_training_or_unfinished_recordings(self):
        for change in ({"training_admitted": True}, {"status": "running"},
                       {"control_authority": "planner"}):
            original = dict(self.record)
            self.record.update(change)
            self.save_completion()
            with self.assertRaisesRegex(ValueError, "completed, non-training"):
                archive_camera_recording(self.root, compact=True)
            self.assertTrue(all(p.exists() for p in self.paths))
            self.record = original

    def test_changed_source_and_unbound_payload_are_not_removed(self):
        self.paths[0].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "not bound"):
            archive_camera_recording(self.root, compact=True)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_corrupt_archive_cannot_remove_or_restore_payloads(self):
        archive_camera_recording(self.root)
        (self.root / ARCHIVE).write_bytes(b"corrupted zip")
        with self.assertRaisesRegex(ValueError, "file identity"):
            archive_camera_recording(self.root, compact=True)
        with self.assertRaisesRegex(ValueError, "file identity"):
            restore_camera_recording(self.root)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_restore_refuses_to_overwrite_conflicting_data(self):
        archive_camera_recording(self.root, compact=True)
        self.paths[0].write_bytes(b"new unrelated bytes")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            restore_camera_recording(self.root)
        self.assertEqual(self.paths[0].read_bytes(), b"new unrelated bytes")

    def test_symlink_payload_is_rejected(self):
        self.paths[0].unlink()
        self.paths[0].symlink_to(self.paths[1])
        with self.assertRaisesRegex(ValueError, "symlink"):
            archive_camera_recording(self.root, compact=True)

    def test_missing_completed_payload_blocks_initial_archival(self):
        self.paths[0].unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            archive_camera_recording(self.root, compact=True)
        self.assertTrue(self.paths[1].exists())


if __name__ == "__main__":
    unittest.main()
