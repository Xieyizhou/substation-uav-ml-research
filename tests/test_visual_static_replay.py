import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ml.artifacts import file_sha256, write_json
from src.ml.visual_benchmark import VisualBenchmarkCondition
from src.ml.visual_static_replay import materialize_static_replay
from src.ml.visual_static_runtime import timing_summary
from src.ml.visual_static_source import (
    ordered_replay_sources,
    static_predict_options,
    write_source_list,
)
from src.ml.visual_yolo_dataset import _write_jsonl
from tests.test_visual_identity import dataset_identity
from tests.test_visual_yolo_package import VisualYoloPackageTests


class StaticReplayTests(unittest.TestCase):
    def test_ordered_source_is_batch_one_and_preserves_membership_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            rows = []
            for name in ("z.png", "a.png", "m.png"):
                (images / name).write_bytes(name.encode())
                rows.append({"image_relative_path": f"images/{name}"})
            with ordered_replay_sources(rows, root) as (source_root, paths):
                source = write_source_list(source_root, "ordered", paths)
                self.assertEqual([path.name for path in paths], [
                    "00000000.png", "00000001.png", "00000002.png",
                ])
                self.assertEqual(
                    [Path(line).name for line in source.read_text().splitlines()],
                    ["00000000.png", "00000001.png", "00000002.png"],
                )
                self.assertEqual(
                    [Path(line).read_bytes().decode() for line in source.read_text().splitlines()],
                    ["z.png", "a.png", "m.png"],
                )
        condition = type("Condition", (), {
            "input_width": 320, "confidence_threshold": 0.65,
        })()
        options = static_predict_options(condition)
        self.assertEqual(options["batch"], 1)
        self.assertFalse(options["rect"])

    def test_timing_summary_reports_exact_count_and_percentiles(self):
        result = timing_summary([1, 2, 3, 4, 5])
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["p50_ms"], 3.0)
        self.assertEqual(result["p95_ms"], 5.0)

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
