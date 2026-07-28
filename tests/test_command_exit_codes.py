import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.logging import analyze_astar_log, compare_experiment_sets, summarize_experiments


class AnalysisExitCodeTests(unittest.TestCase):
    def test_missing_log_returns_failure(self):
        with (
            patch.object(analyze_astar_log, "parse_args", return_value=SimpleNamespace(log=None)),
            patch.object(analyze_astar_log, "ensure_output_tree"),
            patch.object(
                analyze_astar_log,
                "resolve_log_path",
                side_effect=FileNotFoundError("no logs"),
            ),
        ):
            self.assertEqual(analyze_astar_log.main(), 2)


class ComparisonExitCodeTests(unittest.TestCase):
    def args(self, allow_partial=False):
        return SimpleNamespace(
            min_runs_per_stage=3,
            mode="aggregate",
            allow_partial=allow_partial,
            strategy="latest-complete",
        )

    def test_incomplete_comparison_returns_failure(self):
        with (
            patch.object(compare_experiment_sets, "parse_args", return_value=self.args()),
            patch.object(compare_experiment_sets, "ensure_output_tree"),
            patch.object(compare_experiment_sets, "find_all_valid_runs", return_value=[]),
            patch.object(compare_experiment_sets, "print_stage_counts"),
            patch.object(compare_experiment_sets, "write_aggregate_outputs"),
        ):
            self.assertEqual(compare_experiment_sets.main(), 2)

    def test_allow_partial_keeps_success_exit_code(self):
        with (
            patch.object(
                compare_experiment_sets,
                "parse_args",
                return_value=self.args(allow_partial=True),
            ),
            patch.object(compare_experiment_sets, "ensure_output_tree"),
            patch.object(compare_experiment_sets, "find_all_valid_runs", return_value=[]),
            patch.object(compare_experiment_sets, "print_stage_counts"),
            patch.object(compare_experiment_sets, "write_aggregate_outputs"),
        ):
            self.assertEqual(compare_experiment_sets.main(), 0)


class RuntimeStatusEvaluationTests(unittest.TestCase):
    def test_failed_runtime_status_overrides_good_path_metrics(self):
        status = summarize_experiments.choose_status(
            raw_collision=False,
            inflated_entry=False,
            final_error=0.1,
            run_status={"status": "failed", "landing_confirmed": True},
        )
        self.assertEqual(status, "FAIL")

    def test_confirmed_completion_keeps_metric_evaluation(self):
        status = summarize_experiments.choose_status(
            raw_collision=False,
            inflated_entry=False,
            final_error=0.1,
            run_status={"status": "completed", "landing_confirmed": True},
        )
        self.assertEqual(status, "PASS")


if __name__ == "__main__":
    unittest.main()
