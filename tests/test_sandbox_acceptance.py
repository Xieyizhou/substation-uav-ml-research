from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli import sandbox
from src.inspection.config import InspectionConfig
from src.inspection.acceptance import acceptance_summary
from src.sandbox.acceptance import (
    acceptance_snapshot,
    inspect_acceptance,
    run_acceptance,
)
from src.sandbox.supervisor_gate import inspect_supervisor_gate, run_supervisor_gate


HASH = "a" * 64
COMMIT = "b" * 40


def evidence(passed=True):
    return {"passed": passed, "identity": HASH, "path": "evidence.json"}


class SandboxAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        collection = self.root / "collection"
        collection.mkdir()
        plan = collection / "collection_plan.json"
        plan.write_text('{"scenarios": []}\n')
        self.config = InspectionConfig(self.root, plan, collection, self.root / "PX4")

    def tearDown(self):
        self.temporary.cleanup()

    def test_supervisor_gate_exercises_stop_and_failure_diagnostics(self):
        output = self.root / "outputs/sandbox/supervisor_gate/test"
        result = run_supervisor_gate(self.root, output)
        self.assertTrue(result["safe_stop"]["passed"])
        self.assertTrue(result["failure_diagnostic"]["passed"])
        self.assertEqual(
            inspect_supervisor_gate(output / "supervisor_gate.json"), result
        )

    def test_inspection_reports_incomplete_without_evidence(self):
        result = acceptance_summary(self.config)
        self.assertEqual(result["acceptance"]["status"], "incomplete")
        self.assertEqual(result["workflows"], [])

    @patch("src.sandbox.acceptance.latest_challenge_receipt")
    @patch("src.sandbox.acceptance._supervisor_evidence")
    @patch("src.sandbox.acceptance._lidar_evidence")
    @patch("src.sandbox.acceptance._visual_evidence")
    def test_snapshot_requires_every_gate(
        self, visual, lidar, supervisor, challenge
    ):
        visual.return_value = evidence()
        lidar.return_value = (evidence(), evidence(False))
        supervisor.return_value = (evidence(), evidence())
        challenge.return_value = {
            "passed": True, "current": True,
            "challenge_receipt_identity_sha256": HASH,
        }
        result = acceptance_snapshot(self.config)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["closed_loop_flight"]["passed"])

    @patch("src.sandbox.acceptance.acceptance_snapshot")
    @patch("src.sandbox.acceptance.run_supervisor_gate")
    @patch("src.sandbox.acceptance.git_commit", return_value=COMMIT)
    def test_acceptance_binds_evidence_and_is_inspectable(
        self, _commit, supervisor, snapshot
    ):
        checks = {
            name: evidence() for name in (
                "visual_replay", "lidar_replay", "capability_challenge",
                "closed_loop_flight",
                "safe_process_stop", "failure_diagnostics",
            )
        }
        snapshot.return_value = {"passed": True, "checks": checks}
        output = self.root / "outputs/sandbox/acceptance/test"
        result = run_acceptance(self.root, output)
        supervisor.assert_called_once()
        self.assertTrue(result["passed"])
        self.assertIn("not synchronized sensor fusion", result["scope_note"])
        self.assertEqual(inspect_acceptance(output / "acceptance.json"), result)

    @patch("src.sandbox.acceptance.git_commit", return_value="dirty-dirty")
    def test_acceptance_requires_clean_commit(self, _commit):
        with self.assertRaisesRegex(ValueError, "clean tracked commit"):
            run_acceptance(
                self.root, self.root / "outputs/sandbox/acceptance/dirty"
            )

    def test_cli_exposes_acceptance_and_supervisor_inspection(self):
        parser = sandbox.build_parser()
        self.assertEqual(
            parser.parse_args(["acceptance-run", "--output", "out"]).command,
            "acceptance-run",
        )
        self.assertEqual(
            parser.parse_args([
                "supervisor-gate-inspect", "--input", "receipt.json",
            ]).command,
            "supervisor-gate-inspect",
        )
        challenge = parser.parse_args([
            "challenge-run", "--model-id", "model-1",
        ])
        self.assertEqual(challenge.command, "challenge-run")
        self.assertEqual(challenge.model_id, "model-1")


if __name__ == "__main__":
    unittest.main()
