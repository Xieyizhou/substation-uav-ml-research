import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection.runtime import ProcessRecord
from src.ml.artifacts import object_sha256
from src.sandbox.development_app_gate import (
    inspect_development_app_gate,
    run_development_app_gate,
)
from src.sandbox.workflow import WorkflowRecipe


COMMIT = "a" * 40


class NoProcesses:
    def processes(self):
        return ()


class LivePX4:
    def processes(self):
        return (ProcessRecord(42, 1, "/opt/px4", ("px4",)),)


class DevelopmentAppGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        version = {
            "sandbox_product_version": "0.1.0",
            "macos_app_version": "0.2.1",
            "operator_api_version": "1.0",
            "gate_schema_version": 2,
        }
        self.write(self.root / "config/sandbox/version.json", version)
        self.job = self.root / "outputs/sandbox/operator/jobs/job-1"
        recipe = WorkflowRecipe(
            "job-1", "flight_smoke", "flight-smoke", "2026-08-15T00:00:00+00:00",
            600.0, COMMIT, "b" * 64, scenario_id="development-s",
        )
        self.write(self.job / "workflow_recipe.json", recipe.to_record())
        receipt = {
            "workflow_receipt_schema_version": 2,
            "job_id": "job-1", "workflow": "flight_smoke",
            "action": "flight-smoke", "state": "complete", "exit_code": 0,
            "failure": None, "stop_requested": False,
        }
        receipt["receipt_identity_sha256"] = object_sha256(receipt)
        self.write(self.job / "workflow_receipt.json", receipt)
        self.summary = self.root / "outputs/sandbox/flight_smoke/run/summary.json"
        self.write(self.summary, {
            "flight_smoke_schema_version": 2,
            "run_type": "sandbox_flight_smoke", "status": "complete",
            "mission_completed": True, "scenario_id": "development-s",
            "dataset_role": "development", "code_commit": COMMIT,
        })

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @patch("src.sandbox.development_app_gate._tracked_clean", return_value=True)
    @patch("src.sandbox.development_app_gate.git_commit", return_value=COMMIT)
    def test_gate_binds_completed_flight_and_detects_tampering(self, _commit, _clean):
        output = self.root / "gate"
        result = run_development_app_gate(
            self.root, self.job / "workflow_receipt.json",
            self.summary, output, NoProcesses(),
        )
        path = output / "development_app_gate.json"
        self.assertTrue(result["passed"])
        self.assertTrue(inspect_development_app_gate(path)["passed"])
        value = json.loads(path.read_text())
        value["passed"] = False
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_development_app_gate(path)

    @patch("src.sandbox.development_app_gate._tracked_clean", return_value=True)
    @patch("src.sandbox.development_app_gate.git_commit", return_value=COMMIT)
    def test_gate_rejects_residual_runtime(self, _commit, _clean):
        result = run_development_app_gate(
            self.root, self.job / "workflow_receipt.json", self.summary,
            self.root / "gate-live", LivePX4(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["runtime_cleanup"]["passed"])


if __name__ == "__main__":
    unittest.main()
