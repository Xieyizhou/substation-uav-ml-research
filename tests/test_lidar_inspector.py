import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.config import InspectionConfig
from src.inspection.lidar import lidar_summary
from src.sandbox.job_commands import build_command
from src.study.registry import ResearchRegistry


HASH = "a" * 64


class LidarInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        collection = self.root / "collection"
        collection.mkdir()
        plan = collection / "collection_plan.json"
        plan.write_text(json.dumps({"scenarios": []}), encoding="utf-8")
        self.config = InspectionConfig(
            self.root, plan, collection, self.root / "PX4", 0.0
        )
        self.model_id = "lidar-candidate-v1"
        self.dataset_id = "lidar-dataset-v1"
        self.write_replay()
        self.registry = ResearchRegistry(
            self.root / "outputs/research/registry.sqlite"
        )
        self.study_id = self.registry.create_study(
            "candidate gate", self.model_id
        )
        matrix = [
            {
                "scenario_id": f"simple-center-{1001 + index}", "map_id": "simple",
                "target_id": "center", "seed": 1001 + index,
                "condition": condition,
            }
            for condition in (
                "geometric_lidar", "geometric_ml_fusion", "ml_lidar"
            )
            for index in range(5)
        ]
        self.runs = self.registry.ensure_runs(
            self.study_id, "closed-loop", matrix
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_replay(self):
        self.write_json(
            self.root / "outputs/research/lidar_replay_gate_v1/replay_gate.json",
            {
                "passed": True, "model_id": self.model_id,
                "dataset_id": self.dataset_id,
                "replay_gate_identity_sha256": HASH,
                "gate_metrics": {
                    "risk_f1": 0.91, "danger_recall": 0.92,
                    "inference_p95_ms": 2.5, "traversability_iou": 0.8,
                },
            },
        )

    def complete_runs(self):
        for index, run in enumerate(self.runs):
            self.registry.record_metrics(run["run_id"], {
                "mission_success": 1, "landing_success": 1,
                "collision_count": 0, "buffer_entry_count": index == 0,
                "safety_failure_count": 0,
                "sensor_healthy_ratio": 0.99 + index * 0.001,
                "inference_p95_ms": 2.0 + index,
                "predicted_danger_sample_count": 1,
                "replan_attempt_count": 1,
                "successful_replan_count": 1,
                "active_replan_count": 1,
            })

    def test_summary_combines_replay_and_closed_loop_without_writes(self):
        self.complete_runs()
        database = self.root / "outputs/research/registry.sqlite"
        before = (database.stat().st_mtime_ns, database.read_bytes())
        value = lidar_summary(self.config)
        after = (database.stat().st_mtime_ns, database.read_bytes())
        self.assertEqual(before, after)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["replay"]["danger_recall"], 0.92)
        self.assertEqual(value["closed_loop"]["completed"], 15)
        self.assertEqual(value["closed_loop"]["mission_success"], 15)
        self.assertEqual(value["closed_loop"]["buffer_entry_count"], 1)
        self.assertTrue(value["closed_loop"]["passed"])
        self.assertNotIn(str(self.root), json.dumps(value))

    def test_incomplete_study_reports_progress_and_blocks_gate(self):
        self.registry.record_metrics(self.runs[0]["run_id"], {
            "mission_success": 1, "landing_success": 1,
            "collision_count": 0, "safety_failure_count": 0,
        })
        closed = lidar_summary(self.config)["closed_loop"]
        self.assertEqual(closed["status"], "in_progress")
        self.assertEqual(closed["completed"], 1)
        self.assertEqual(closed["pending"], 14)
        self.assertFalse(closed["passed"])

    def test_gate_commands_resolve_identity_bound_local_artifacts(self):
        self.write_json(
            self.root / "models/lidar/candidate/model_manifest.json",
            {"model_id": self.model_id},
        )
        self.write_json(
            self.root / "outputs/research/dataset/dataset_manifest.json",
            {"dataset_id": self.dataset_id},
        )
        replay = build_command(self.config, "lidar-replay-gate")
        self.assertIn("replay-gate", replay.argv)
        self.assertIn(str(self.root / "models/lidar/candidate"), replay.argv)
        self.assertIn(str(self.root / "outputs/research/dataset"), replay.argv)
        self.assertIn("outputs/sandbox/lidar_replay", " ".join(replay.argv))
        challenge_command = build_command(self.config, "lidar-challenge-gate")
        self.assertIn("challenge-run", challenge_command.argv)
        self.assertIn(self.model_id, challenge_command.argv)
        challenge = {
            "challenge_receipt_identity_sha256": HASH,
            "path": str(self.root / "challenge_receipt.json"),
        }
        with patch(
            "src.sandbox.lidar_jobs.require_current_challenge",
            return_value=challenge,
        ):
            closed = build_command(self.config, "lidar-closed-loop-next")
        self.assertEqual(closed.argv[-2:], ("--max-runs", "1"))
        self.assertIn(self.study_id, closed.argv)
        self.complete_runs()
        with patch(
            "src.sandbox.lidar_jobs.require_current_challenge",
            return_value=challenge,
        ):
            with self.assertRaisesRegex(ValueError, "no pending runs"):
                build_command(self.config, "lidar-closed-loop-next")


if __name__ == "__main__":
    unittest.main()
