import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ml.artifacts import file_sha256, write_json
from src.ml.visual_benchmark import VisualBenchmarkCondition
from src.ml.visual_static_replay import materialize_static_replay
from src.ml.visual_yolo_dataset import _write_jsonl
from tests.test_visual_identity import dataset_identity
from tests.test_visual_yolo_package import VisualYoloPackageTests


class StaticReplayTests(unittest.TestCase):
    def test_materialization_binds_nine_conditions_and_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, heldout, output = root / "package", root / "heldout", root / "out"
            VisualYoloPackageTests()._package(package)
            package_record = json.loads((package / "manifest.json").read_text())
            identity = dataset_identity(dataset_role="held_out_test")
            write_json(
                heldout / "identity/held_out_test_dataset_identity.json",
                identity.to_record(),
            )
            membership = heldout / "identity/heldout_test_membership.jsonl"
            _write_jsonl(
                membership,
                [{"sample_id": "sample", "image_relative_path": "image.png"}],
            )
            write_json(
                heldout / "identity/heldout_access_receipt.json",
                {
                    "model_package_identity_sha256": package_record[
                        "package_identity_sha256"
                    ],
                    "heldout_access_identity_sha256": "f" * 64,
                    "membership_sha256": file_sha256(membership),
                    "frozen_confidence_threshold": 0.42,
                },
            )
            with patch(
                "src.ml.visual_static_replay._clean_commit",
                return_value="commit",
            ):
                result = materialize_static_replay(
                    package,
                    heldout,
                    "benchmarks/visual_static_v1",
                    output,
                )
            self.assertEqual(len(result["conditions"]), 9)
            identities = set()
            for item in result["conditions"]:
                condition = VisualBenchmarkCondition.from_record(
                    json.loads((output / item["path"]).read_text())
                )
                self.assertEqual(condition.confidence_threshold, 0.42)
                self.assertEqual(condition.measured_frame_count, 1)
                identities.add(condition.condition_identity_sha256)
            self.assertEqual(len(identities), 9)


if __name__ == "__main__":
    unittest.main()
