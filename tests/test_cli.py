"""Tests for the unified project command center."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.cli import astar, experiments, reports
from src.cli.root import build_parser, main


class RootCliTests(unittest.TestCase):
    def test_root_help_lists_every_public_command_group(self):
        help_text = build_parser().format_help()
        for command in (
            "map",
            "point",
            "task",
            "astar",
            "experiment",
            "report",
            "check",
            "maintenance",
        ):
            self.assertIn(command, help_text)

    def test_missing_command_returns_usage_error(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main([])
        self.assertEqual(status, 2)
        self.assertIn("Unified command center", output.getvalue())

    @patch("src.cli.root.run_script", return_value=7)
    def test_forwarded_manager_exit_code_is_preserved(self, run_script):
        status = main(["point", "current"])
        self.assertEqual(status, 7)
        run_script.assert_called_once_with(
            "scripts/maps/switch_target.py",
            ["current"],
            "Opening destination manager",
        )


class ExperimentCliTests(unittest.TestCase):
    @patch("src.cli.experiments.run_shell", return_value=0)
    def test_official_stage_maps_to_one_internal_runner(self, run_shell):
        status = experiments.main(["run", "active-replan"])
        self.assertEqual(status, 0)
        run_shell.assert_called_once_with(
            "scripts/flight/experiments/run_active_replan.sh",
            description="Running experiment: Active local replanning",
        )

    @patch("src.cli.experiments.run_shell", return_value=4)
    def test_run_all_forwards_trial_count_and_exit_code(self, run_shell):
        status = experiments.main(["run-all", "--trials", "5"])
        self.assertEqual(status, 4)
        run_shell.assert_called_once_with(
            "scripts/flight/experiments/run_all_3x.sh",
            ["5"],
            "Running all experiment stages (5 trials each)",
        )


class AdvancedCliTests(unittest.TestCase):
    @patch("src.cli.astar.run_module", return_value=0)
    def test_astar_preview_uses_module_and_dry_run(self, run_module):
        status = astar.main(["preview", "--return-home"])
        self.assertEqual(status, 0)
        module, arguments, _ = run_module.call_args.args
        self.assertEqual(module, "src.flight.fly_astar_path")
        self.assertEqual(arguments[0], "--dry-run")
        self.assertIn("--return-home", arguments)
        self.assertIn("--obstacle-config", arguments)

    @patch("src.cli.astar.run_module", return_value=0)
    def test_legacy_grid_does_not_receive_selected_map_override(self, run_module):
        status = astar.main(["preview", "--use-built-in-grid"])
        self.assertEqual(status, 0)
        arguments = run_module.call_args.args[1]
        self.assertIn("--use-built-in-grid", arguments)
        self.assertNotIn("--obstacle-config", arguments)

    @patch("src.cli.reports.run_module", return_value=0)
    def test_report_compare_forwards_advanced_arguments(self, run_module):
        status = reports.main(["compare", "--mode", "aggregate"])
        self.assertEqual(status, 0)
        run_module.assert_called_once_with(
            "src.logging.compare_experiment_sets",
            ["--mode", "aggregate"],
            "Running report: compare",
        )

    @patch("src.cli.reports.run_module", return_value=0)
    def test_report_collect_maps_forwards_coverage_arguments(self, run_module):
        status = reports.main(
            ["collect-maps", "--min-runs-per-map", "2", "--strict"]
        )
        self.assertEqual(status, 0)
        run_module.assert_called_once_with(
            "src.logging.map_run_collection",
            ["--min-runs-per-map", "2", "--strict"],
            "Running report: collect-maps",
        )


if __name__ == "__main__":
    unittest.main()
