from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.study.closed_loop_worker import _execute_row, _run_one
from src.vision.collection.process import CollectionProcessError


class ClosedLoopOutcomeTests(unittest.TestCase):
    @patch("src.study.closed_loop_worker.ingest_results")
    @patch("src.study.closed_loop_worker._run_one")
    def test_execute_row_materializes_safe_mission_failure(self, run_one, ingest):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "flight.csv"
            log_path.write_text("elapsed_s\n1\n", encoding="utf-8")
            result_path = root / "result.json"
            run_one.return_value = (
                log_path, {"mission_success": 0, "landing_success": 1},
                {
                    "status": "failed", "message": "waypoint timeout",
                    "landing_confirmed": True,
                },
            )
            registry = MagicMock()
            row = {
                "run_id": "run-1", "study_id": "study-1",
                "scenario_id": "scenario-1", "condition": "geometric_lidar",
                "oracle_planner_config": "planner.json",
                "result_path": str(result_path),
            }
            _execute_row(
                registry, row, root / "attempt", "formal", root, None,
                startup_timeout_s=1.0, probe_timeout_s=1.0,
                flight_timeout_s=1.0,
            )
            self.assertTrue(result_path.exists())
        ingest.assert_called_once()

    @patch("src.study.closed_loop_worker.mission_metrics", return_value={"mission_success": 0})
    @patch("src.study.closed_loop_worker.landed_mission_status")
    @patch("src.study.closed_loop_worker._new_flight_log")
    @patch("src.study.closed_loop_worker.wait_for_study_flight")
    @patch("src.study.closed_loop_worker.stop_process")
    @patch("src.study.closed_loop_worker.ensure_process_running")
    @patch("src.study.closed_loop_worker.start_process")
    @patch("src.study.closed_loop_worker._probe_lidar", return_value="/world/test/scan")
    @patch("src.study.closed_loop_worker._run_setup")
    def test_safe_failed_mission_is_an_experiment_result(
        self, setup, probe, start, ensure, stop, wait, new_log, status, metrics
    ):
        wait.side_effect = CollectionProcessError("flight task exited with code 1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_log.return_value = root / "flight.csv"
            status.return_value = {
                "status": "failed", "message": "waypoint timeout",
                "landing_confirmed": True,
            }
            start.side_effect = [object(), object()]
            row = {
                "setup_commands": [], "launcher_environment": {},
                "launcher_command": ["launcher"],
                "flight_command": ["python", "main.py", "task"],
                "oracle_planner_config": "planner.json",
            }
            _, result, mission = _run_one(
                row, root / "run", startup_timeout_s=1.0,
                probe_timeout_s=1.0, flight_timeout_s=1.0,
            )
        self.assertEqual(result["mission_success"], 0)
        self.assertEqual(mission["status"], "failed")
        metrics.assert_called_once()


if __name__ == "__main__":
    unittest.main()
