import json
import tempfile
import unittest
from pathlib import Path

from src.logging.active_replan_validation import validate_active_replan_rows
from src.logging import validate_active_replan_runs as batch_validation


def telemetry_rows(sequence=("RWP03", "RWP03", "RWP04", "RWP05")):
    rows = [
        {
            "elapsed_s": 0.0,
            "phase": "outbound_to_goal",
            "route_direction": "outbound",
            "target_name": "WP06",
            "replan_mode": "active",
            "replan_route_replaced": False,
            "active_replan_count": 0,
        },
        {
            "elapsed_s": 1.0,
            "phase": "outbound_to_goal",
            "route_direction": "outbound",
            "target_name": "WP06",
            "replan_mode": "active",
            "replan_route_replaced": True,
            "active_replan_count": 1,
        },
    ]
    for offset, name in enumerate(sequence, start=2):
        final = offset == len(sequence) + 1
        rows.append(
            {
                "elapsed_s": float(offset),
                "phase": "outbound_to_goal",
                "route_direction": "outbound",
                "target_name": name,
                "target_north_m": 10.0 if final else float(offset),
                "target_east_m": 20.0 if final else float(offset),
                "target_down_m": -1.5,
                "horizontal_error_m": 0.2 if final else 1.0,
                "error_down_m": 0.1,
                "replan_mode": "active",
                "replan_route_replaced": False,
                "active_replan_count": 1,
            }
        )
    rows.extend(
        [
            {
                "elapsed_s": 10.0,
                "phase": "goal_hover",
                "route_direction": "none",
                "target_name": "WP10",
                "target_north_m": 10.0,
                "target_east_m": 20.0,
                "target_down_m": -1.5,
                "horizontal_error_m": 0.2,
                "error_down_m": 0.1,
                "replan_mode": "active",
                "active_replan_count": 1,
            },
            {
                "elapsed_s": 11.0,
                "phase": "return_to_start",
                "route_direction": "return",
                "target_name": "WP09",
                "replan_mode": "active",
                "active_replan_count": 1,
            },
            {
                "elapsed_s": 12.0,
                "phase": "landed",
                "route_direction": "none",
                "target_name": "WP01",
                "replan_mode": "active",
                "active_replan_count": 1,
            },
        ]
    )
    return rows


class ActiveReplanValidationTests(unittest.TestCase):
    def test_runtime_status_can_confirm_landing_after_telemetry_stops(self):
        rows = telemetry_rows()
        rows = [row for row in rows if row["phase"] != "landed"]

        result = validate_active_replan_rows(
            rows,
            run_status={"status": "completed", "landing_confirmed": True},
        )

        self.assertEqual(result["active_replan_target_switching_status"], "PASS")
        self.assertTrue(result["mission_completed"])

    def test_failed_runtime_status_cannot_override_missing_landing(self):
        rows = telemetry_rows()
        rows = [row for row in rows if row["phase"] != "landed"]

        result = validate_active_replan_rows(
            rows,
            run_status={"status": "failed", "landing_confirmed": True},
        )

        self.assertEqual(result["active_replan_target_switching_status"], "FAIL")
        self.assertFalse(result["mission_completed"])

    def test_valid_contiguous_sequence_collapses_repeated_samples(self):
        result = validate_active_replan_rows(telemetry_rows())
        self.assertEqual(result["active_replan_target_switching_status"], "PASS")
        self.assertEqual(result["post_replan_unique_target_sequence"], ["RWP03", "RWP04", "RWP05"])
        self.assertEqual(result["post_replan_old_wp_target_count"], 0)

    def test_valid_sequence_may_start_later_than_rwp01(self):
        result = validate_active_replan_rows(telemetry_rows(("RWP07", "RWP08")))
        self.assertEqual(result["active_replan_target_switching_status"], "PASS")
        self.assertEqual(result["first_replanned_target_name"], "RWP07")

    def test_old_wp_reappearing_after_replacement_fails(self):
        rows = telemetry_rows()
        rows.insert(4, {**rows[3], "elapsed_s": 3.5, "target_name": "WP09"})
        result = validate_active_replan_rows(rows)
        self.assertEqual(result["active_replan_target_switching_status"], "FAIL")
        self.assertEqual(result["post_replan_old_wp_target_count"], 1)

    def test_numerical_gap_fails(self):
        result = validate_active_replan_rows(telemetry_rows(("RWP03", "RWP05")))
        self.assertEqual(result["active_replan_target_switching_status"], "FAIL")
        self.assertFalse(result["rwp_sequence_contiguous"])

    def test_backward_transition_fails(self):
        result = validate_active_replan_rows(telemetry_rows(("RWP06", "RWP05")))
        self.assertEqual(result["active_replan_target_switching_status"], "FAIL")

    def test_missing_transition_evidence_is_unavailable(self):
        rows = telemetry_rows()
        for row in rows:
            row.pop("replan_route_replaced", None)
            row.pop("active_replan_count", None)
        result = validate_active_replan_rows(rows)
        self.assertEqual(result["active_replan_target_switching_status"], "UNAVAILABLE")

    def test_non_active_replan_run_is_not_applicable(self):
        rows = telemetry_rows()
        for row in rows:
            row["replan_mode"] = "log_only"
        result = validate_active_replan_rows(rows)
        self.assertEqual(result["active_replan_target_switching_status"], "NOT_APPLICABLE")

    def test_original_goal_not_reached_fails(self):
        rows = telemetry_rows()
        goal_hover = next(row for row in rows if row["phase"] == "goal_hover")
        goal_hover["horizontal_error_m"] = 0.8
        result = validate_active_replan_rows(rows)
        self.assertEqual(result["original_goal_reached"], False)
        self.assertEqual(result["active_replan_target_switching_status"], "FAIL")


class ActiveReplanBatchTests(unittest.TestCase):
    def write_run(self, root, run_id, status):
        run_dir = root / run_id
        run_dir.mkdir()
        manifest = {
            "run_id": run_id,
            "replan_summary": {"replan_mode": "active"},
            "active_replan_target_validation": {
                "pre_replan_target_name": "WP06",
                "first_replanned_target_name": "RWP03",
                "post_replan_unique_target_sequence": ["RWP03", "RWP04"],
                "post_replan_old_wp_target_count": 0,
                "original_goal_reached": True,
                "active_replan_target_switching_status": status,
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

    def test_three_run_batch_all_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                self.write_run(root, f"as_20260713_00000{index}", "PASS")
            selected, eligible, _, passed = batch_validation.evaluate_latest_runs(root, 3)
            self.assertEqual(len(selected), 3)
            self.assertEqual(len(eligible), 3)
            self.assertTrue(passed)

    def test_three_run_batch_one_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, status in enumerate(("PASS", "FAIL", "PASS")):
                self.write_run(root, f"as_20260713_00000{index}", status)
            self.assertFalse(batch_validation.evaluate_latest_runs(root, 3)[3])

    def test_fewer_than_three_eligible_runs_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                self.write_run(root, f"as_20260713_00000{index}", "PASS")
            selected, eligible, _, passed = batch_validation.evaluate_latest_runs(root, 3)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(eligible), 2)
            self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
