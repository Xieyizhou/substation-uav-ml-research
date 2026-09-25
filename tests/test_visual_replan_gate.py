from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from src.sandbox.job_commands import build_command
from src.sandbox.managed_flight_stop import cooperative_stop
from src.sandbox.live_replan_gate import BASE
from src.sandbox.visual_replan_gate import summary, require_visual_gate


class VisualGateTests(unittest.TestCase):
    def test_missing_evidence_disables_and_rejects_launch(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(summary(root)["ready_for_launch_revalidation"])
            with self.assertRaises(OSError):
                require_visual_gate(root)

    def test_command_is_fixed_and_uses_visual_gate(self):
        config = SimpleNamespace(profile="development", project_root=Path("/tmp"))
        with patch("src.sandbox.visual_replan_gate.require_visual_gate", side_effect=ValueError("stale evidence")):
            with self.assertRaisesRegex(ValueError, "stale evidence"):
                build_command(config, "visual-replan-flight")
        with patch("src.sandbox.visual_replan_gate.require_visual_gate", return_value={"positive_receipt_sha256": "a"*64, "stop_receipt_sha256": "b"*64}):
            command = build_command(config, "visual-replan-flight")
            self.assertIn("src.sandbox.visual_replan_entry", command.argv)
            self.assertEqual(len(command.expected_outputs), 3)
            for parameters in ({"speed": 1}, {"model": "unknown"}):
                with self.assertRaisesRegex(ValueError, "no overrides"):
                    build_command(config, "visual-replan-flight", parameters=parameters)
            config.profile = "formal"
            with self.assertRaises(ValueError):
                build_command(config, "visual-replan-flight")

    def test_visual_stop_uses_same_scoped_marker_without_mutating_job(self):
        with tempfile.TemporaryDirectory() as root:
            run = "sandbox-replan-v1-"+"e"*32
            job = SimpleNamespace(action="visual-replan-flight", scenario_id=run, job_id="job", error=None)
            self.assertTrue(cooperative_stop(SimpleNamespace(project_root=Path(root)), job, lambda: True))
            self.assertEqual(job.action, "visual-replan-flight")
            self.assertTrue((Path(root)/BASE/run/"stop-requested.json").exists())

    def test_visual_job_keeps_landing_window_on_app_shutdown(self):
        from src.sandbox.operator import SandboxOperator
        operator = SandboxOperator.__new__(SandboxOperator)
        operator._guard = threading.RLock()
        operator._active = SimpleNamespace(action="visual-replan-flight", job_id="owned")
        operator._thread, operator.stop = Mock(), Mock()
        operator.shutdown()
        operator._thread.join.assert_called_once_with(timeout=130.)


if __name__ == "__main__":
    unittest.main()
