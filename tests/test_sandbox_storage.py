from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.cli.sandbox import build_parser, main as sandbox_main
from src.inspection.config import InspectionConfig
from src.sandbox.failure_classification import classify_failure
from src.sandbox.job_models import SandboxJob, SandboxJobStore, utc_now
from src.sandbox.retention import (
    apply_retention_plan,
    create_retention_plan,
    inspect_retention_plan,
)
from src.sandbox.storage_policy import (
    GIB,
    OutputBudgetExceeded,
    capture_output_baseline,
    output_budget_violation,
    require_output_budget,
    storage_summary,
)


class SandboxStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        collection = self.root / "collection"
        collection.mkdir()
        plan = collection / "collection_plan.json"
        plan.write_text('{"scenarios": []}', encoding="utf-8")
        self.config = InspectionConfig(
            self.root, plan, collection, self.root / "PX4", 5.0,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _old_experiments(self, count=12):
        root = self.root / "outputs/sandbox/experiments"
        timestamp = datetime(2026, 8, 10, tzinfo=timezone.utc).timestamp()
        for index in range(count):
            path = root / f"run-{index:02d}"
            path.mkdir(parents=True)
            (path / "result.json").write_text(f'{{"run": {index}}}')
            os.utime(path / "result.json", (timestamp, timestamp))
            os.utime(path, (timestamp, timestamp))
        return root

    def test_storage_summary_is_limited_to_sandbox_outputs(self):
        sandbox = self.root / "outputs/sandbox/experiments/run/result.bin"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_bytes(b"x" * 10)
        research = self.root / "data/research/large.bin"
        research.parent.mkdir(parents=True)
        research.write_bytes(b"x" * 100)

        result = storage_summary(self.config)

        self.assertEqual(result["sandbox_output_bytes"], 10)
        self.assertEqual(result["groups"], [{"name": "experiments", "bytes": 10}])

    def test_storage_cli_is_offline_and_confirmation_is_required(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["storage"]).command, "storage")
        output = self.root / "outputs/sandbox/retention/plan.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sandbox_main([
                "--project-root", str(self.root), "retention-plan",
                "--output", str(output),
            ]), 0)
            self.assertEqual(sandbox_main([
                "--project-root", str(self.root), "retention-apply",
                "--input", str(output), "--confirm-identity", "wrong",
            ]), 1)

    def test_output_budget_blocks_before_reserved_space_is_consumed(self):
        usage = shutil.disk_usage(self.root)
        constrained = type(usage)(usage.total, usage.total - 6 * GIB, 6 * GIB)
        with patch("src.sandbox.storage_policy.shutil.disk_usage", return_value=constrained):
            with self.assertRaisesRegex(OutputBudgetExceeded, "disk budget"):
                require_output_budget(self.config, "collection-gate")
            doctor = require_output_budget(self.config, "doctor")
        self.assertTrue(doctor["passed"])

    def test_tracked_output_budget_ignores_unrelated_disk_decline(self):
        output = self.root / "outputs/sandbox/flight_smoke"
        output.mkdir(parents=True)
        baseline = capture_output_baseline(
            self.config, ("outputs/sandbox/flight_smoke",)
        )
        job = SandboxJob(
            "flight", "flight-smoke", "running", utc_now(), 60.0,
            output_budget_bytes=10, disk_free_bytes_at_start=1000,
            disk_reserve_bytes=100, output_baseline_bytes=baseline,
        )
        usage = shutil.disk_usage(self.root)
        changed = type(usage)(usage.total, usage.total - 500, 500)
        with patch(
            "src.sandbox.storage_policy.shutil.disk_usage", return_value=changed
        ):
            self.assertIsNone(output_budget_violation(self.config, job))
            (output / "run.log").write_bytes(b"x" * 11)
            self.assertIn(
                "output allowance", output_budget_violation(self.config, job)
            )

    def test_retention_plan_keeps_newest_and_requires_explicit_apply(self):
        root = self._old_experiments()
        output = self.root / "outputs/sandbox/retention/plan.json"
        result = create_retention_plan(
            self.config, output,
            now=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )

        plan = inspect_retention_plan(output)
        self.assertEqual(plan["candidate_bytes"], result["plan"]["candidate_bytes"])
        self.assertEqual(
            [item["path"] for item in plan["candidates"]],
            [
                "outputs/sandbox/experiments/run-10",
                "outputs/sandbox/experiments/run-11",
            ],
        )
        self.assertTrue((root / "run-10").is_dir())
        applied = apply_retention_plan(self.config, output)
        self.assertEqual(applied["removed_count"], 2)
        self.assertFalse((root / "run-10").exists())
        self.assertTrue((root / "run-09").is_dir())

    def test_retention_rejects_changed_candidate_and_active_job(self):
        root = self._old_experiments()
        output = self.root / "outputs/sandbox/retention/plan.json"
        create_retention_plan(
            self.config, output,
            now=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )
        (root / "run-10/result.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "candidate changed"):
            apply_retention_plan(self.config, output)

        (root / "run-10/result.json").write_text('{"run": 10}')
        create_retention_plan(
            self.config, output,
            now=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )
        store = SandboxJobStore(self.config.sandbox_jobs_root)
        store.write(SandboxJob("active", "doctor", "running", utc_now(), 5.0))
        with self.assertRaisesRegex(ValueError, "job is active"):
            apply_retention_plan(self.config, output)

    def test_formal_profile_never_produces_prune_candidates(self):
        self._old_experiments()
        config = InspectionConfig(
            self.config.project_root, self.config.plan_path,
            self.config.collection_root, self.config.px4_root,
            self.config.minimum_free_gib, "formal",
        )
        output = self.root / "outputs/sandbox/retention/formal.json"
        plan = create_retention_plan(config, output)["plan"]
        self.assertTrue(plan["formal_retention_locked"])
        self.assertEqual(plan["candidates"], [])
        with self.assertRaisesRegex(ValueError, "cannot be pruned"):
            apply_retention_plan(config, output)

    def test_retention_rejects_symlinked_group_root(self):
        outside = self.root / "outside"
        outside.mkdir()
        group = self.root / "outputs/sandbox/experiments"
        group.parent.mkdir(parents=True)
        group.symlink_to(outside, target_is_directory=True)
        output = self.root / "outputs/sandbox/retention/plan.json"
        with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
            create_retention_plan(self.config, output)

    def test_failure_classification_is_stable_and_structured(self):
        job = SandboxJob("timeout", "flight", "failed", utc_now(), 5.0)
        job.error = "job exceeded 5s timeout"
        classify_failure(job)
        self.assertEqual((job.failure_code, job.failure_retryable), (
            "deadline_exceeded", True,
        ))
        job.stop_requested = True
        classify_failure(job)
        self.assertEqual(job.failure_code, "user_cancelled")
        job.stop_requested = False
        job.error = "PX4 telemetry connection unavailable"
        classify_failure(job)
        self.assertEqual(job.failure_code, "runtime_unavailable")

    def test_retention_plan_identity_detects_tampering(self):
        output = self.root / "outputs/sandbox/retention/plan.json"
        create_retention_plan(self.config, output)
        value = json.loads(output.read_text())
        value["minimum_age_hours"] = 0
        output.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_retention_plan(output)


if __name__ == "__main__":
    unittest.main()
