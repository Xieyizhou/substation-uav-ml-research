import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModuleBoundaryTests(unittest.TestCase):
    def line_count(self, relative_path):
        path = PROJECT_ROOT / relative_path
        return len(path.read_text(encoding="utf-8").splitlines())

    def test_core_files_stay_below_one_thousand_lines(self):
        for path in (
            "src/flight/fly_astar_path.py",
            "src/flight/flight_config.py",
            "src/logging/analyze_astar_log.py",
            "src/logging/summarize_experiments.py",
            "src/logging/compare_experiment_sets.py",
            "src/logging/plotting.py",
            "src/logging/report_writer.py",
        ):
            with self.subTest(path=path):
                self.assertLessEqual(self.line_count(path), 1000)

    def test_extracted_modules_stay_below_five_hundred_lines(self):
        for path in (
            "src/flight/flight_state.py",
            "src/flight/flight_cli.py",
            "src/flight/flight_defaults.py",
            "src/flight/flight_planner_config.py",
            "src/flight/flight_runtime_config.py",
            "src/flight/landing_manager.py",
            "src/flight/mavsdk_preflight.py",
            "src/flight/mission_lifecycle.py",
            "src/flight/perception_response.py",
            "src/flight/replanning_controller.py",
            "src/flight/route_planning.py",
            "src/flight/run_status.py",
            "src/flight/runtime_settings.py",
            "src/flight/telemetry_runtime.py",
            "src/flight/waypoint_executor.py",
            "src/logging/analysis_inference.py",
            "src/logging/analysis_summaries.py",
            "src/logging/analysis_warnings.py",
            "src/logging/comparison_aggregate.py",
            "src/logging/comparison_collection.py",
            "src/logging/comparison_landmark.py",
            "src/logging/comparison_schema.py",
            "src/logging/plot_diagnostics.py",
            "src/logging/plot_timeseries.py",
            "src/logging/plot_trajectory.py",
            "src/logging/report_files.py",
            "src/logging/report_sections.py",
            "src/logging/report_summary.py",
            "src/logging/summary_collection.py",
            "src/logging/summary_outputs.py",
            "src/logging/summary_values.py",
            "src/vision/contracts/annotations.py",
            "src/vision/contracts/benchmark.py",
            "src/vision/contracts/identity.py",
            "src/vision/replay/benchmark_matrix.py",
            "src/sensors/camera_decoded.py",
            "src/sensors/camera_decoder.py",
        ):
            with self.subTest(path=path):
                self.assertLessEqual(
                    self.line_count(path),
                    500,
                )

    def test_all_python_implementation_files_stay_below_five_hundred_lines(self):
        paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
        paths.extend(sorted((PROJECT_ROOT / "scripts").rglob("*.py")))
        for path in paths:
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    500,
                )

    def test_visual_domain_files_stay_below_three_hundred_lines(self):
        paths = sorted((PROJECT_ROOT / "src" / "vision").rglob("*.py"))
        for path in paths:
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    300,
                )


if __name__ == "__main__":
    unittest.main()
