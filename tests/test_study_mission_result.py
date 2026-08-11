from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.ml.artifacts import write_json
from src.study.mission_result import (
    landed_mission_status,
    mission_metrics,
    result_payload,
)
from src.vision.collection.process import CollectionProcessError


class StudyMissionResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_status_accepts_safe_failure_and_requires_landing(self):
        log_path = self.root / "flight.csv"
        status_path = log_path.with_suffix(".status.json")
        write_json(status_path, {
            "status": "failed", "message": "waypoint timeout",
            "landing_confirmed": True,
        })
        self.assertEqual(landed_mission_status(log_path)["status"], "failed")
        write_json(status_path, {"status": "failed", "landing_confirmed": False})
        with self.assertRaisesRegex(CollectionProcessError, "confirmed landing"):
            landed_mission_status(log_path)

    @patch("src.study.mission_result.lidar_quality_metrics", return_value={})
    @patch("src.study.mission_result.obstacle_collision_report")
    @patch("src.study.mission_result.build_obstacle_map", return_value={})
    @patch("src.study.mission_result.replan_summary", return_value={})
    @patch("src.study.mission_result.perception_summary")
    @patch("src.study.mission_result.prepare_dataframe")
    def test_metrics_preserve_failed_mission_as_zero(
        self, prepare, perception, replan, obstacle_map, collision, quality
    ):
        frame = MagicMock()
        frame.empty = False
        frame.__getitem__.return_value.max.return_value = 12.5
        prepare.return_value = frame
        perception.return_value = {"sensor_healthy_ratio": 1.0}
        collision.return_value = {
            "raw_physical_collision_detected": False,
            "inflated_safety_buffer_entry_detected": True,
        }
        planner = self.root / "planner.json"
        write_json(planner, {"resolution_m": 1.0})

        metrics = mission_metrics(
            self.root / "flight.csv", planner,
            {"status": "failed", "landing_confirmed": True},
        )

        self.assertEqual(metrics["mission_success"], 0)
        self.assertEqual(metrics["landing_success"], 1)
        self.assertEqual(metrics["buffer_entry_count"], 1)

    @patch("src.study.mission_result.git_commit", return_value="abc123")
    @patch("src.study.mission_result.file_sha256", return_value="f" * 64)
    def test_result_payload_records_terminal_mission(self, sha256, commit):
        payload = result_payload(
            {"run_id": "run-1", "scenario_id": "scenario-1", "condition": "ml"},
            self.root / "flight.csv", {"mission_success": 0},
            {
                "status": "failed", "message": "waypoint timeout",
                "landing_confirmed": True,
            },
            {"formal_study_identity_sha256": "a" * 64},
        )
        self.assertEqual(payload["mission"]["status"], "failed")
        self.assertEqual(payload["metrics"]["mission_success"], 0)
        self.assertEqual(payload["formal_study_identity_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
