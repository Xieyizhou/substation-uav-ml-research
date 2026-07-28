import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.logging.summary_collection import collect_run


class SummaryCollectionTests(unittest.TestCase):
    def test_collect_run_normalizes_manifest_summary_and_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "astar_fixture.csv"
            with log_path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "elapsed_s",
                        "map_name",
                        "local_north_m",
                        "local_east_m",
                        "target_name",
                        "target_north_m",
                        "target_east_m",
                        "route_direction",
                        "horizontal_error_m",
                        "perception_risk_level",
                        "replan_triggered",
                        "replan_success",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "elapsed_s": 0,
                            "map_name": "substation_complex_v1",
                            "local_north_m": 0,
                            "local_east_m": 0,
                            "target_name": "WP01",
                            "target_north_m": 3,
                            "target_east_m": 4,
                            "route_direction": "outbound",
                            "horizontal_error_m": 5,
                            "perception_risk_level": "clear",
                            "replan_triggered": "false",
                            "replan_success": "false",
                        },
                        {
                            "elapsed_s": 2.5,
                            "map_name": "substation_complex_v1",
                            "local_north_m": 1.5,
                            "local_east_m": 2,
                            "target_name": "WP01",
                            "target_north_m": 3,
                            "target_east_m": 4,
                            "route_direction": "outbound",
                            "horizontal_error_m": 2.5,
                            "perception_risk_level": "warning",
                            "replan_triggered": "false",
                            "replan_success": "false",
                        },
                        {
                            "elapsed_s": 5,
                            "map_name": "substation_complex_v1",
                            "local_north_m": 3,
                            "local_east_m": 4,
                            "target_name": "WP02",
                            "target_north_m": 6,
                            "target_east_m": 8,
                            "route_direction": "outbound",
                            "horizontal_error_m": 0.2,
                            "perception_risk_level": "danger",
                            "replan_triggered": "true",
                            "replan_success": "true",
                        },
                    ]
                )
            run_dir = root / "outputs" / "04_active_replan" / "runs" / "as_fixture"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "as_fixture",
                        "source_log": str(log_path),
                        "map_name": "substation_complex_v1",
                        "run_status": {
                            "status": "completed",
                            "landing_confirmed": True,
                        },
                        "collision_report": {
                            "raw_physical_collision_detected": False,
                            "inflated_safety_buffer_entry_detected": False,
                        },
                    }
                )
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "stage": "active_replan",
                        "experiment_type": "active_local_replan",
                    }
                )
            )
            (run_dir / "summary.md").write_text("- Final horizontal error: 0.2 m\n")
            row = collect_run(run_dir)

        self.assertEqual(row["map_name"], "substation_complex_v1")
        self.assertEqual(row["stage"], "active_replan")
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["completed_or_failed"], "completed")
        self.assertAlmostEqual(row["duration_s"], 5.0)
        self.assertAlmostEqual(row["actual_traveled_distance_m"], 5.0)
        self.assertAlmostEqual(row["planned_path_length_m"], 5.0)
        self.assertEqual(row["perception_risk_detection_count"], 2)


if __name__ == "__main__":
    unittest.main()
