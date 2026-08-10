import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ml.artifacts import object_sha256, write_json
from src.study.formal_spec import freeze_formal_study, load_formal_spec
from src.study.registry import ResearchRegistry


class FormalStudySpecificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = ResearchRegistry(self.root / "registry.sqlite")
        manifest = {
            "model_id": "candidate",
            "onnx_sha256": "a" * 64,
            "dataset_id": "dataset",
            "dataset_sha256": "b" * 64,
            "parent_model": None,
        }
        self.registry.register_model(manifest, self.root / "model")
        self.study = self.registry.create_study("formal", "candidate")
        self.replay = {
            "replay_gate_identity_sha256": "c" * 64,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_specification_is_the_frozen_four_condition_protocol(self):
        value = load_formal_spec()
        self.assertEqual(len(value["conditions"]), 4)
        self.assertEqual(value["bootstrap"]["samples"], 2000)

    @patch("src.study.formal_spec.git_commit", return_value="commit")
    def test_receipt_is_identity_bound_and_idempotent(self, _commit):
        first = freeze_formal_study(
            self.registry, self.study, self.root / "results", self.replay,
            qualification_study_id="qualification-study",
        )
        second = freeze_formal_study(
            self.registry, self.study, self.root / "results", self.replay,
            qualification_study_id="qualification-study",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["qualification_study_id"], "qualification-study")
        self.assertEqual(first["flight_timeout_policy"]["max_timeout_s"], 480.0)
        supplied = first.pop("formal_study_identity_sha256")
        self.assertEqual(supplied, object_sha256(first))

    @patch("src.study.formal_spec.git_commit", return_value="commit-dirty")
    def test_dirty_code_cannot_freeze_formal_evidence(self, _commit):
        with self.assertRaisesRegex(ValueError, "clean tracked commit"):
            freeze_formal_study(
                self.registry, self.study, self.root / "results", self.replay
            )


if __name__ == "__main__":
    unittest.main()
