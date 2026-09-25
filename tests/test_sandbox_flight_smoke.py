import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.sandbox.flight_smoke import (
    _flight_command,
    _mission_completed,
    choose_scenario,
    run_flight_smoke,
)
from src.vision.collection.display import launcher_environment


class FlightSmokeTests(unittest.TestCase):
    def plan(self):
        return {
            "scenarios": [{
                "scenario_id": "development-smoke",
                "recording_id": "recording-smoke",
                "dataset_role": "development",
                "split": "development",
            }]
        }

    def test_command_redirects_events_outside_formal_recording(self):
        prepared = {
            "flight_command": [
                "python", "main.py", "task", "run", "fly_round_trip", "--",
                "--visual-mission-events", "formal/events.jsonl",
            ]
        }
        command = _flight_command(prepared, Path("smoke/events.jsonl"))
        self.assertEqual(command[-1], "smoke/events.jsonl")

    def test_blind_scenario_is_rejected(self):
        plan = self.plan()
        plan["scenarios"][0]["dataset_role"] = "blind"
        with self.assertRaisesRegex(ValueError, "blind"):
            choose_scenario(plan, ".", "development-smoke")

    def test_display_mode_only_changes_window_policy(self):
        prepared = {"launcher_environment": {"HEADLESS": "1", "WORLD": "same"}}
        self.assertEqual(
            launcher_environment(prepared, "headless"),
            {"HEADLESS": "1", "WORLD": "same", "UAV_SANDBOX_DISPLAY_MODE": "headless"},
        )
        self.assertEqual(
            launcher_environment(prepared, "visual_preview"),
            {"WORLD": "same", "UAV_SANDBOX_DISPLAY_MODE": "visual_preview"},
        )
        self.assertEqual(prepared["launcher_environment"]["HEADLESS"], "1")

    def test_actual_shell_policy_overrides_inherited_headless(self):
        script = Path("scripts/flight/start_px4_substation.sh").read_text()
        policy = script.split('DISPLAY_MODE="', 1)[1].split('[[ -d "$PROJECT_ROOT" ]]', 1)[0]
        policy = 'DISPLAY_MODE="' + policy
        for mode, expected in (("visual_preview", "unset"), ("headless", "1")):
            env = dict(os.environ, HEADLESS="1")
            env.update(launcher_environment({"launcher_environment": {}}, mode))
            result = subprocess.run(
                ["bash", "-c", 'set -eu\nfail() { exit 2; }\n' + policy +
                 '\nprintf "%s" "${HEADLESS-unset}"'],
                env=env, text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout, expected)

    def test_invalid_display_mode_rejected(self):
        with self.assertRaises(ValueError):
            launcher_environment({"launcher_environment": {}}, "unknown")

    def test_completion_requires_confirmed_landing_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text(json.dumps({
                "event_type": "mission_completed",
                "landing_confirmed": True,
            }) + "\n")
            self.assertTrue(_mission_completed(path))

    @patch("src.sandbox.flight_smoke.git_commit", return_value="commit")
    @patch("src.sandbox.flight_smoke.stop_process")
    @patch("src.sandbox.flight_smoke.wait_process")
    @patch("src.sandbox.flight_smoke.start_process")
    @patch("src.sandbox.flight_smoke.probe_until_ready")
    @patch("src.sandbox.flight_smoke.ensure_process_running")
    @patch("src.sandbox.flight_smoke.prepare_collection_scenario")
    @patch("src.sandbox.flight_smoke.load_collection_plan")
    def test_success_writes_non_dataset_summary(
        self, load_plan, prepare, ensure, probe, start, wait, stop, commit
    ):
        load_plan.return_value = self.plan()
        prepare.return_value = {
            "launcher_environment": {},
            "launcher_command": ["launcher"],
            "flight_command": [
                "python", "flight", "--visual-mission-events", "formal.jsonl",
            ],
            "flight_timeout_s": 10.0,
        }
        start.side_effect = [
            SimpleNamespace(process=Mock(), log_path=Path("simulator.log")),
            SimpleNamespace(process=Mock(), log_path=Path("flight.log")),
        ]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.sandbox.flight_smoke._mission_completed", return_value=True
        ):
            root = run_flight_smoke(
                "plan.json", temporary, temporary,
                scenario_id="development-smoke",
            )
            summary = json.loads((root / "summary.json").read_text())
        self.assertEqual(summary["run_type"], "sandbox_flight_smoke")
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["mission_completed"])
        self.assertEqual(summary["display_mode"], "headless")
        self.assertEqual(start.call_count, 2)


if __name__ == "__main__":
    unittest.main()
