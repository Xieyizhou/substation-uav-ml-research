import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.logging import analyze_astar_log


class AnalysisPipelineTests(unittest.TestCase):
    def test_main_builds_summary_manifest_and_metadata_from_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "astar_20260727_120000.csv"
            with log.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "elapsed_s",
                        "planner_name",
                        "map_name",
                        "local_north_m",
                        "local_east_m",
                        "local_down_m",
                        "horizontal_error_m",
                        "altitude_m",
                        "perception_enabled",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "elapsed_s": 0,
                        "planner_name": "astar_grid",
                        "map_name": "synthetic",
                        "local_north_m": 0,
                        "local_east_m": 0,
                        "local_down_m": -1,
                        "horizontal_error_m": 0.2,
                        "altitude_m": 1,
                        "perception_enabled": False,
                    }
                )
            status = log.with_suffix(".status.json")
            status.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "landing_confirmed": True,
                        "phase": "landed",
                    }
                )
            )
            run_dir = root / "run"
            collision = {
                "raw_physical_collision_detected": False,
                "inflated_safety_buffer_entry_detected": False,
                "obstacle_collision_detected": False,
                "raw_collision_points": [],
                "inflated_buffer_entry_points": [],
                "first_raw_collision_timestamps": [],
                "first_inflated_buffer_entry_timestamps": [],
                "raw_obstacle_names_involved": [],
                "inflated_obstacle_names_involved": [],
                "approximate_min_clearance_m": None,
            }
            args = SimpleNamespace(log=log, obstacle_config=None, debug_plots=False)
            with (
                patch.object(analyze_astar_log, "parse_args", return_value=args),
                patch.object(analyze_astar_log, "ensure_output_tree"),
                patch.object(analyze_astar_log, "get_run_output_dir", return_value=run_dir),
                patch.object(
                    analyze_astar_log,
                    "load_analysis_obstacles",
                    return_value=(None, None, [], None, None),
                ),
                patch.object(
                    analyze_astar_log,
                    "obstacle_collision_report",
                    return_value=collision,
                ),
                patch.object(
                    analyze_astar_log,
                    "compute_warnings",
                    return_value=([], [], collision),
                ),
                patch.object(analyze_astar_log, "save_trajectory_plot", return_value=None),
                patch.object(analyze_astar_log, "save_error_plot", return_value=None),
                patch.object(
                    analyze_astar_log,
                    "save_perception_risk_timeline",
                    return_value=None,
                ),
            ):
                result = analyze_astar_log.main()

            self.assertEqual(result, 0)
            self.assertTrue((run_dir / "summary.md").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "run_metadata.json").is_file())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["run_status"]["phase"], "landed")
            self.assertEqual(manifest["map_name"], "synthetic")


if __name__ == "__main__":
    unittest.main()
