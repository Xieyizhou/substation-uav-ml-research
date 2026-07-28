import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.logging import report_writer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReportWriterModuleTests(unittest.TestCase):
    def test_compatibility_module_preserves_public_writers(self):
        expected = {
            "generated_file_description",
            "markdown_generated_file_list",
            "markdown_perception_summary",
            "markdown_replan_summary",
            "markdown_transition_table",
            "save_collision_points_csv",
            "write_manifest",
            "write_run_metadata",
            "write_summary",
        }
        self.assertTrue(expected.issubset(set(report_writer.__all__)))
        for name in expected:
            self.assertTrue(callable(getattr(report_writer, name)))

    def test_summary_wrapper_keeps_section_renderers_patchable(self):
        frame = pd.DataFrame(
            {
                "elapsed_s": [0.0, 1.0],
                "planner_name": ["astar_grid"] * 2,
                "map_name": ["synthetic"] * 2,
                "altitude_m": [1.0, 1.0],
                "horizontal_error_m": [1.0, 0.2],
            }
        )
        collision_report = {
            "raw_physical_collision_detected": False,
            "inflated_safety_buffer_entry_detected": False,
            "raw_collision_points": [],
            "inflated_buffer_entry_points": [],
            "first_raw_collision_timestamps": [],
            "first_inflated_buffer_entry_timestamps": [],
            "raw_obstacle_names_involved": [],
            "inflated_obstacle_names_involved": [],
            "approximate_min_clearance_m": None,
        }
        target_validation = {
            "post_replan_unique_target_sequence": [],
            "pre_replan_target_name": None,
            "first_replanned_target_name": None,
            "first_replanned_target_elapsed_s": None,
            "post_replan_old_wp_target_count": 0,
            "rwp_sequence_contiguous": None,
            "original_goal_reached": None,
            "mission_completed": None,
            "active_replan_target_switching_status": "NOT_APPLICABLE",
            "active_replan_target_switching_notes": "synthetic",
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(
                report_writer,
                "markdown_transition_table",
                return_value=["compatibility renderer used"],
            ):
                result = report_writer.write_summary(
                    output_dir=output_dir,
                    log_path=PROJECT_ROOT / "synthetic.csv",
                    run_id="as_synthetic",
                    created_at_utc="2026-07-20T00:00:00+00:00",
                    df=frame,
                    core_generated_files=[],
                    debug_generated_files=[],
                    collision_generated_files=[],
                    debug_plots_enabled=False,
                    obstacle_config_path=None,
                    collision_report=collision_report,
                    warnings=[],
                    obstacle_map=None,
                    waypoint_transition_summary_func=lambda data: [],
                    perception_summary_func=lambda data: {"available": False},
                    replan_summary_func=lambda data: None,
                    active_replan_route_replacement_summary_func=lambda data: None,
                    active_replan_target_validation_func=lambda data: target_validation,
                    infer_return_home_enabled_func=lambda data: False,
                    waypoint_reached_threshold_m=0.4,
                )
            self.assertIn("compatibility renderer used", result.read_text())

    def test_summary_handles_missing_optional_data_and_surfaces_warnings(self):
        frame = pd.DataFrame({"elapsed_s": [0.0], "map_name": ["edge_case"]})
        collision = {
            "raw_physical_collision_detected": False,
            "inflated_safety_buffer_entry_detected": False,
            "raw_collision_points": [],
            "inflated_buffer_entry_points": [],
            "first_raw_collision_timestamps": [],
            "first_inflated_buffer_entry_timestamps": [],
            "raw_obstacle_names_involved": [],
            "inflated_obstacle_names_involved": [],
            "approximate_min_clearance_m": None,
        }
        validation = {
            "active_replan_target_switching_status": "NOT_APPLICABLE",
            "active_replan_target_switching_notes": "no active replan",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = report_writer.write_summary(
                output_dir=Path(directory),
                log_path=PROJECT_ROOT / "missing.csv",
                run_id="as_edge",
                created_at_utc="2026-07-27T00:00:00+00:00",
                df=frame,
                core_generated_files=[],
                debug_generated_files=[],
                collision_generated_files=[],
                debug_plots_enabled=False,
                obstacle_config_path=None,
                collision_report=collision,
                warnings=["synthetic warning"],
                obstacle_map=None,
                waypoint_transition_summary_func=lambda data: [],
                perception_summary_func=lambda data: {"available": False},
                replan_summary_func=lambda data: None,
                active_replan_route_replacement_summary_func=lambda data: None,
                active_replan_target_validation_func=lambda data: validation,
                infer_return_home_enabled_func=lambda data: False,
                waypoint_reached_threshold_m=0.4,
            )
            text = path.read_text()
        self.assertIn("synthetic warning", text)
        self.assertIn("legacy/unavailable", text)
        self.assertIn("Debug plots were not generated", text)


if __name__ == "__main__":
    unittest.main()
