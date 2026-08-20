import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.cli.studies import build_parser, main
from src.study.closed_loop_worker import execute_speed_envelope
from src.study.matrix import tier_matrix
from src.study.mission_result import _event_latency_ms
from src.study.runner import _flight_arguments
from src.study.speed_envelope import speed_envelope_matrix, speed_envelope_report


def _passing_rows():
    return [
        {
            "scenario_id": definition["scenario_id"],
            "metrics": {
                "configured_speed_m_s": definition["speed_m_s"],
                "mission_success": 1,
                "route_switch_correct": 1,
                "landing_success": 1,
                "event_chain_complete": 1,
                "collision_count": 0,
                "safety_failure_count": 0,
                "detection_to_resume_ms": 750.0,
            },
        }
        for definition in speed_envelope_matrix()
    ]


class SpeedEnvelopeTests(unittest.TestCase):
    def test_cli_exposes_execution_and_frozen_report_commands(self):
        execute = build_parser().parse_args([
            "execute-speed-envelope", "study-v1", "--max-runs", "1",
            "--scenario-id", "speed-simple-early-0p5-r01",
        ])
        report = build_parser().parse_args([
            "speed-envelope-report", "study-v1",
        ])
        self.assertEqual(execute.command, "execute-speed-envelope")
        self.assertEqual(execute.max_runs, 1)
        self.assertEqual(report.command, "speed-envelope-report")

    @patch("src.study.closed_loop_worker.execute_flight_tier")
    def test_worker_executes_the_speed_envelope_tier(self, execute_tier):
        execute_speed_envelope(
            "registry.sqlite", "study-v1", "results", max_runs=1
        )
        execute_tier.assert_called_once_with(
            "registry.sqlite", "study-v1", "results",
            tier="speed-envelope", max_runs=1,
        )

    @patch("src.cli.studies.speed_envelope_report")
    @patch("src.cli.studies.ResearchRegistry")
    def test_cli_report_reads_only_speed_envelope_metrics(
        self, registry_type, report
    ):
        registry = registry_type.return_value
        registry.run_metrics.return_value = []
        report.return_value = {"passed": True}
        with redirect_stdout(io.StringIO()):
            return_code = main([
                "--registry", "registry.sqlite",
                "speed-envelope-report", "study-v1",
            ])
        self.assertEqual(return_code, 0)
        registry.run_metrics.assert_called_once_with(
            "study-v1", "speed-envelope"
        )
        report.assert_called_once_with([])

    def test_matrix_is_the_frozen_four_by_five_by_three_design(self):
        rows = speed_envelope_matrix()
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["scenario_id"] for row in rows}), 60)
        self.assertEqual(len({row["representative_scenario"] for row in rows}), 4)
        self.assertEqual(
            {row["speed_m_s"] for row in rows}, {0.5, 0.75, 1.0, 1.25, 1.5}
        )
        self.assertTrue(all(row["repeat"] in {1, 2, 3} for row in rows))
        self.assertEqual(tier_matrix("speed-envelope"), rows)

    def test_speed_arguments_apply_to_outbound_and_return_legs(self):
        arguments = _flight_arguments(
            "geometric_lidar", "model.onnx", "scenario.json",
            "planner.json", "dynamic.json", 1.25,
        )
        self.assertEqual(arguments[arguments.index("--max-speed") + 1], "1.25")
        self.assertEqual(
            arguments[arguments.index("--return-speed-scale") + 1], "1.0"
        )

    def test_report_recommends_the_highest_contiguous_passing_speed(self):
        report = speed_envelope_report(_passing_rows())
        self.assertTrue(report["passed"])
        self.assertEqual(report["recommended_max_speed_m_s"], 1.5)

    def test_report_stops_at_the_first_unsafe_speed(self):
        rows = _passing_rows()
        for row in rows:
            if row["metrics"]["configured_speed_m_s"] == 1.5:
                row["metrics"]["collision_count"] = 1
                break
        report = speed_envelope_report(rows)
        self.assertTrue(report["passed"])
        self.assertEqual(report["recommended_max_speed_m_s"], 1.25)
        self.assertFalse(report["speed_reports"][-1]["passed"])

    def test_incomplete_matrix_cannot_issue_a_recommendation(self):
        report = speed_envelope_report(_passing_rows()[:-1])
        self.assertFalse(report["passed"])
        self.assertIsNone(report["recommended_max_speed_m_s"])

    def test_missing_latency_evidence_fails_the_affected_speed(self):
        rows = _passing_rows()
        rows[0]["metrics"]["detection_to_resume_ms"] = None
        report = speed_envelope_report(rows)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["recommended_max_speed_m_s"])

    def test_event_latency_uses_ordered_monotonic_timestamps(self):
        events = [
            {
                "event_type": "dynamic_blocker_detected",
                "host_monotonic_ns": 1_000_000_000,
            },
            {
                "event_type": "dynamic_replan_resumed",
                "host_monotonic_ns": 1_125_000_000,
            },
        ]
        self.assertEqual(
            _event_latency_ms(
                events, "dynamic_blocker_detected", "dynamic_replan_resumed"
            ),
            125.0,
        )


if __name__ == "__main__":
    unittest.main()
