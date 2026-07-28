import unittest
from unittest.mock import patch

from src.flight import task_runner
from src.flight.task_presets import TASKS


class FlightTaskTests(unittest.TestCase):
    def test_required_compact_tasks_exist(self):
        self.assertTrue(
            {
                "preview_route",
                "fly_to_point",
                "fly_round_trip",
                "fly_with_perception",
                "replan_log_only",
                "fly_with_replan",
            }.issubset(TASKS)
        )

    def test_advanced_arguments_override_after_preset(self):
        args = task_runner.task_arguments(
            "fly_to_point",
            ["--max-speed", "0.6"],
        )
        self.assertEqual(args[-2:], ["--max-speed", "0.6"])

    def test_task_runner_calls_existing_engine_in_process(self):
        with (
            patch.object(task_runner, "print_task_summary"),
            patch.object(task_runner, "managed_task_runtime") as runtime,
            patch.object(task_runner, "flight_main", return_value=None) as flight_main,
        ):
            status = task_runner.run_task("preview_route")
        self.assertEqual(status, 0)
        expected = ["--dry-run", "--compact-output"]
        flight_main.assert_called_once_with(expected)
        runtime.assert_called_once_with(expected)

    def test_experiment_task_presets_match_expected_modes(self):
        perception = TASKS["fly_with_perception"]["args"]
        log_only = TASKS["replan_log_only"]["args"]
        active = TASKS["fly_with_replan"]["args"]
        self.assertIn("slow_down", perception)
        self.assertIn("log_only", log_only)
        self.assertIn("active", active)


if __name__ == "__main__":
    unittest.main()
