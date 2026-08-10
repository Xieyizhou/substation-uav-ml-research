"""Read-only Sandbox v1 acceptance and workflow receipt summaries."""

from __future__ import annotations

import json

from src.sandbox.acceptance import acceptance_snapshot, inspect_acceptance
from src.sandbox.workflow import WorkflowRecipe, inspect_workflow


def _relative(root, path):
    return path.resolve().relative_to(root.resolve()).as_posix()


def _latest_acceptance(config):
    paths = sorted(
        config.project_root.glob("outputs/sandbox/acceptance/*/acceptance.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not paths:
        snapshot = acceptance_snapshot(config)
        return {
            "status": "ready" if snapshot["passed"] else "incomplete",
            "passed": False,
            "checks": snapshot["checks"],
        }
    path = paths[0]
    try:
        value = inspect_acceptance(path)
        return {
            "status": "complete" if value["passed"] else "failed",
            "passed": value["passed"],
            "identity": value["acceptance_result_identity_sha256"],
            "commit": value["software_commit_sha"],
            "checks": value["checks"],
            "scope_note": value["scope_note"],
            "path": _relative(config.project_root, path),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"status": "invalid", "passed": False, "error": str(error)}


def _workflow_summary(config, directory):
    recipe = WorkflowRecipe.from_record(json.loads(
        (directory / "workflow_recipe.json").read_text(encoding="utf-8")
    ))
    receipt = inspect_workflow(
        directory / "workflow_receipt.json", "receipt_identity_sha256"
    )
    if receipt["recipe_identity_sha256"] != recipe.recipe_identity_sha256:
        raise ValueError("workflow receipt references a different recipe")
    return {
        "job_id": receipt["job_id"],
        "workflow": receipt["workflow"],
        "action": receipt["action"],
        "state": receipt["state"],
        "started_at": receipt["started_at"],
        "ended_at": receipt["ended_at"],
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "outputs": receipt["outputs"],
        "diagnostics": receipt["diagnostics"],
    }


def _workflow_summaries(config):
    root = config.sandbox_jobs_root
    if not root.is_dir():
        return []
    rows = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not (directory / "workflow_receipt.json").is_file():
            continue
        try:
            rows.append(_workflow_summary(config, directory))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rows.append({
                "job_id": directory.name,
                "state": "invalid",
                "error": str(error),
            })
    return rows[:50]


def acceptance_summary(config):
    return {
        "acceptance": _latest_acceptance(config),
        "workflows": _workflow_summaries(config),
    }
