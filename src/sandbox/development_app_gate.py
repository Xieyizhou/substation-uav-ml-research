"""Evidence gate for one App-managed Development flight smoke."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from src.inspection.runtime import LocalProcessAdapter, runtime_status
from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.sandbox.gate_outcome import check_outcome, validate_outcome
from src.sandbox.version import load_sandbox_version
from src.sandbox.workflow import WorkflowRecipe, inspect_workflow


DEVELOPMENT_APP_GATE_SCHEMA_VERSION = 1
BLIND_ROLES = frozenset({"blind", "held_out_test", "test"})


def _tracked_clean(root):
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _read_recipe(receipt_path):
    path = Path(receipt_path).with_name("workflow_recipe.json")
    return WorkflowRecipe.from_record(json.loads(path.read_text(encoding="utf-8")))


def _check(passed, reason_code, detail):
    return check_outcome(passed, None if passed else reason_code, detail)


def _gate_checks(root, receipt, recipe, summary, runtime, summary_path, commit):
    scenario_matches = bool(recipe.scenario_id) and (
        recipe.scenario_id == summary.get("scenario_id")
    )
    return {
        "managed_job": _check(
            receipt.get("workflow") == "flight_smoke"
            and receipt.get("action") == "flight-smoke"
            and receipt.get("state") == "complete"
            and receipt.get("exit_code") == 0
            and receipt.get("failure") is None
            and receipt.get("stop_requested") is False,
            "managed_job_incomplete", "App-managed flight job completed",
        ),
        "flight_summary": _check(
            summary.get("flight_smoke_schema_version") == 2
            and summary.get("run_type") == "sandbox_flight_smoke"
            and summary.get("status") == "complete"
            and summary.get("mission_completed") is True
            and scenario_matches,
            "flight_summary_invalid", "flight summary confirms mission completion",
        ),
        "non_blind_boundary": _check(
            str(summary.get("dataset_role", "")).lower() not in BLIND_ROLES,
            "blind_scenario_forbidden", "flight smoke used a non-blind scenario",
        ),
        "software_identity": _check(
            recipe.software_commit_sha == commit
            and summary.get("code_commit") == commit,
            "software_identity_mismatch", "source identities match",
        ),
        "runtime_cleanup": _check(
            all(item.available and not item.alive for item in runtime),
            "runtime_cleanup_failed", "no simulator task remains",
        ),
        "no_frame_payloads": _check(
            not any(summary_path.parent.rglob("*.png")),
            "unexpected_frame_payload", "flight smoke created no PNG payloads",
        ),
        "tracked_worktree": _check(
            _tracked_clean(root), "tracked_worktree_dirty", "tracked worktree is clean",
        ),
    }


def run_development_app_gate(
    project_root, workflow_receipt, flight_summary, output, process_adapter=None,
):
    root = Path(project_root).resolve()
    receipt_path = Path(workflow_receipt).resolve()
    summary_path = Path(flight_summary).resolve()
    receipt = inspect_workflow(receipt_path, "receipt_identity_sha256")
    recipe = _read_recipe(receipt_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime = runtime_status(process_adapter or LocalProcessAdapter())
    commit = git_commit(root)
    checks = _gate_checks(
        root, receipt, recipe, summary, runtime, summary_path, commit,
    )
    record = {
        "development_app_gate_schema_version": DEVELOPMENT_APP_GATE_SCHEMA_VERSION,
        "versions": load_sandbox_version(root).to_record(),
        "software_commit_sha": commit,
        "scenario_id": recipe.scenario_id,
        "job_id": receipt.get("job_id"),
        "workflow_receipt_sha256": file_sha256(receipt_path),
        "flight_summary_sha256": file_sha256(summary_path),
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }
    record["development_app_gate_identity_sha256"] = object_sha256(record)
    write_json(Path(output) / "development_app_gate.json", record)
    return record


def inspect_development_app_gate(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("development_app_gate_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("development App gate identity mismatch")
    if value.get("development_app_gate_schema_version") != DEVELOPMENT_APP_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported development App gate schema")
    for check in value.get("checks", {}).values():
        validate_outcome(check)
    passed = bool(value.get("checks")) and all(
        check.get("passed") is True for check in value["checks"].values()
    )
    if value.get("passed") is not passed:
        raise ValueError("development App gate status is inconsistent")
    return {**value, "development_app_gate_identity_sha256": supplied}
