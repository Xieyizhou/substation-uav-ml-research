import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.config import InspectionConfig
from src.inspection.research import (
    BLIND_RESULT,
    PACKAGE_ROOT,
    REPLAY_416_REPEAT,
    REPLAY_ROOT,
    TRAINING_IDENTITY,
    research_summary,
)


HASH = "a" * 64
CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")


class ResearchInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        collection = self.root / "collection"
        collection.mkdir()
        plan = collection / "collection_plan.json"
        plan.write_text(json.dumps({"scenarios": []}))
        self.config = InspectionConfig(
            self.root, plan, collection, self.root / "PX4", 0.0
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def metric(self, value):
        return {
            "standard_metrics": {
                "precision": value,
                "recall": value,
                "mAP50": value,
                "mAP50_95": value,
            },
            "frozen_threshold_metrics": {
                "macro_f1": value,
                "small_object_recall": value,
                "no_target_false_positive_rate": 0.01,
                "per_class": {name: {"recall": value} for name in CLASSES},
            },
        }

    def replay(self, latency=10.0):
        return {
            "completion_status": "completed",
            "frame_counts": {"inferred": 10, "skipped": 0},
            "resource_metrics": {"throughput_fps": 30.0},
            "visual_metrics": {
                "precision": 0.9,
                "recall": 0.8,
                "small_object_recall": 0.7,
            },
            "timing_summaries": {
                "end_to_end_ms": {
                    "p50_ms": latency,
                    "p95_ms": latency + 1,
                    "p99_ms": latency + 2,
                }
            },
        }

    def test_missing_artifacts_have_explicit_non_complete_states(self):
        result = research_summary(self.config)
        self.assertEqual(result["complete_stage_count"], 0)
        self.assertEqual(result["training_view"]["status"], "missing")
        self.assertEqual(result["paired_blind"]["status"], "sealed")
        self.assertEqual(result["static_replay"]["status"], "partial")

    def test_complete_summary_uses_controlled_416_latency_replicate(self):
        self.write(
            TRAINING_IDENTITY,
            {
                "training_view_identity_sha256": HASH,
                "sampling_algorithm": "balanced-v2",
                "train_frame_count": 10,
                "validation_frame_count": 4,
                "full_validation_frame_count": 20,
            },
        )
        self.write(PACKAGE_ROOT / "manifest.json", {})
        blind = {
            "frame_count": 30,
            "evaluation_code_commit_sha": "commit",
            "results": {"v1": self.metric(0.2), "v2": self.metric(0.9)},
            "comparison": {
                "mAP50_95_delta_v2_minus_v1": 0.7,
                "macro_f1_delta_v2_minus_v1": 0.7,
                "paired_bootstrap": {
                    "macro_f1_delta_ci95": [0.6, 0.8],
                    "probability_v2_improves": 1.0,
                },
            },
        }
        self.write(BLIND_RESULT, blind)
        for size in (320, 416, 640):
            for policy in ("every-frame", "every-2nd", "every-3rd"):
                self.write(
                    REPLAY_ROOT / "results" / f"visual-static-v1-{size}-{policy}.json",
                    self.replay(180.0 if (size, policy) == (416, "every-frame") else 10.0),
                )
        self.write(REPLAY_416_REPEAT, self.replay(14.0))
        package = {
            "package_identity_sha256": HASH,
            "architecture": "yolo11n",
            "frozen_confidence_threshold": 0.37,
            "exports": {"320": {}, "416": {}, "640": {}},
        }
        with patch(
            "src.inspection.research.validate_yolo_package",
            return_value=package,
        ):
            result = research_summary(self.config)
        self.assertEqual(result["complete_stage_count"], 4)
        self.assertEqual(result["paired_blind"]["v2"]["macro_f1"], 0.9)
        row = next(
            item
            for item in result["static_replay"]["rows"]
            if item["input_size"] == 416 and item["policy"] == "every_frame"
        )
        self.assertEqual(row["source"], "controlled_replicate")
        self.assertEqual(row["p50_ms"], 14.0)
        self.assertNotIn("predictions", result["paired_blind"])


if __name__ == "__main__":
    unittest.main()
