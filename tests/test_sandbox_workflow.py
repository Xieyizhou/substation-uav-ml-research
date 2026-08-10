import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.sandbox.job_commands import SandboxCommand
from src.sandbox.job_models import SandboxJob, SandboxJobStore, utc_now
from src.sandbox.workflow import (
    WorkflowArtifact,
    WorkflowRecipe,
    inspect_workflow,
    materialize_workflow_receipt,
    materialize_workflow_recipe,
)


class SandboxWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = SandboxJobStore(self.root / "outputs/sandbox/operator/jobs")
        self.job = SandboxJob("job-1", "doctor", "complete", utc_now(), 5.0)
        self.job.started_at = utc_now()
        self.job.ended_at = utc_now()
        self.job.exit_code = 0

    def tearDown(self):
        self.temporary.cleanup()

    @patch("src.sandbox.workflow.git_commit", return_value="a" * 40)
    def test_materializes_identity_bound_recipe_and_receipt(self, _commit):
        output = self.root / "outputs/sandbox/example/result.json"
        output.parent.mkdir(parents=True)
        output.write_text('{"passed": true}\n', encoding="utf-8")
        command = SandboxCommand(
            "doctor", (sys.executable, "main.py", "sandbox", "doctor"), 5.0,
            workflow="environment_check",
            artifacts=(WorkflowArtifact("config", "fixed", "config/example.json"),),
            expected_outputs=("outputs/sandbox/example/result.json",),
            requires_runtime_idle=False,
        )
        self.store.write(self.job)
        self.store.log_path(self.job.job_id).write_text("complete\n", encoding="utf-8")
        recipe = materialize_workflow_recipe(
            self.root, self.store, self.job, command
        )
        receipt = materialize_workflow_receipt(
            self.root, self.store, self.job, recipe
        )
        loaded = WorkflowRecipe.from_record(json.loads(
            (self.store.directory("job-1") / "workflow_recipe.json").read_text()
        ))
        self.assertEqual(loaded, recipe)
        self.assertEqual(receipt["state"], "complete")
        self.assertTrue(receipt["outputs"][0]["present"])
        self.assertEqual(
            inspect_workflow(
                self.store.directory("job-1") / "workflow_receipt.json",
                "receipt_identity_sha256",
            )["receipt_identity_sha256"],
            receipt["receipt_identity_sha256"],
        )

    def test_recipe_and_receipt_tampering_is_rejected(self):
        recipe = WorkflowRecipe(
            "job-1", "check", "doctor", utc_now(), 5.0, "unknown",
            "a" * 64,
        )
        record = recipe.to_record()
        record["action"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            WorkflowRecipe.from_record(record)
        receipt = {"state": "complete"}
        path = self.root / "receipt.json"
        path.write_text(json.dumps({
            **receipt, "receipt_identity_sha256": "b" * 64,
        }))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_workflow(path, "receipt_identity_sha256")


if __name__ == "__main__":
    unittest.main()
