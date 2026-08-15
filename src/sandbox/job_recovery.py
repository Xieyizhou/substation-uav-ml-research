"""Persistence helpers for adopting an interrupted sandbox operator job."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time

from src.sandbox.job_models import utc_now
from src.sandbox.failure_classification import classify_failure
from src.sandbox.workflow import WorkflowRecipe, materialize_workflow_receipt
from src.sandbox.storage_policy import output_budget_violation


def read_process_result(store, job_id):
    path = store.process_result_path(job_id)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("process_result_schema_version") != 1:
        raise ValueError("unsupported managed process result schema")
    if not isinstance(value.get("exit_code"), int) or not value.get("ended_at"):
        raise ValueError("managed process result is incomplete")
    return value


def read_workflow_recipe(store, job_id):
    path = store.directory(job_id) / "workflow_recipe.json"
    return WorkflowRecipe.from_record(json.loads(path.read_text(encoding="utf-8")))


def elapsed_seconds(job):
    if not job.started_at:
        return 0.0
    started = datetime.fromisoformat(job.started_at)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def recovered_budget_violation(config, job, next_check):
    now = time.monotonic()
    if now < next_check:
        return None, next_check
    return output_budget_violation(config, job), now + 2.0


def apply_process_result(job, value):
    token = value.get("ownership_token")
    if job.ownership_token and token != job.ownership_token:
        raise ValueError("managed process result ownership mismatch")
    job.exit_code = value["exit_code"]
    job.ended_at = value["ended_at"]
    if job.exit_code == 0 and not job.stop_requested and job.error is None:
        job.state = "complete"
    else:
        job.state = "failed"
        if job.error is None:
            job.error = (
                "job stopped by request" if job.stop_requested
                else f"job exited with code {job.exit_code}"
            )


def append_diagnostics(store, job):
    if job.state != "failed" or job.sensitive:
        return
    try:
        lines = store.log_path(job.job_id).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return
    noteworthy = [
        line.strip() for line in lines
        if any(word in line.lower() for word in ("error", "failed", "timeout"))
    ]
    job.diagnostics.extend(noteworthy[-5:])


def finalize_recovered(project_root, store, job):
    append_diagnostics(store, job)
    classify_failure(job)
    store.write(job)
    try:
        recipe = read_workflow_recipe(store, job.job_id)
        materialize_workflow_receipt(project_root, store, job, recipe)
    except Exception as error:
        job.state, job.ended_at = "failed", job.ended_at or utc_now()
        job.error = f"workflow receipt failed: {error}"
        classify_failure(job)
        store.write(job)
