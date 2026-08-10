"""Evidence-bound Sandbox v1 acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path

from src.inspection.config import InspectionConfig
from src.inspection.lidar import lidar_summary
from src.ml.artifacts import git_commit, object_sha256, write_json
from src.sandbox.experiment_runner import inspect_result
from src.sandbox.supervisor_gate import inspect_supervisor_gate, run_supervisor_gate


ACCEPTANCE_RECIPE_SCHEMA_VERSION = 1
ACCEPTANCE_RESULT_SCHEMA_VERSION = 1
MINIMUM_VISUAL_FRAMES = 64


def _latest_valid(paths, inspect):
    error = None
    for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            return path, inspect(path)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as current:
            error = str(current)
    return None, error


def _visual_evidence(root):
    path, inspected = _latest_valid(
        root.glob("outputs/sandbox/experiments/*/result.json"), inspect_result
    )
    if path is None:
        return {"passed": False, "error": inspected or "visual replay result is missing"}
    result = inspected["result"]
    frames = int(result.get("frame_counts", {}).get("inferred", 0))
    return {
        "passed": frames >= MINIMUM_VISUAL_FRAMES,
        "identity": result["result_identity_sha256"],
        "path": path.relative_to(root).as_posix(),
        "summary": {
            "inferred_frames": frames,
            "precision": result.get("metrics", {}).get("precision"),
            "recall": result.get("metrics", {}).get("recall"),
            "p95_ms": result.get("timing", {}).get("end_to_end_ms", {}).get("p95_ms"),
        },
    }


def _lidar_evidence(config):
    value = lidar_summary(config)
    replay, closed = value["replay"], value["closed_loop"]
    return (
        {
            "passed": bool(replay.get("passed")),
            "identity": replay.get("identity"),
            "path": replay.get("path"),
            "summary": {
                "macro_f1": replay.get("macro_f1"),
                "danger_recall": replay.get("danger_recall"),
                "p95_ms": replay.get("inference_p95_ms"),
            },
        },
        {
            "passed": bool(closed.get("passed")) and closed.get("completed") == closed.get("total"),
            "identity": object_sha256(closed),
            "path": closed.get("path"),
            "summary": {
                "completed": closed.get("completed"),
                "total": closed.get("total"),
                "mission_success": closed.get("mission_success"),
                "landing_success": closed.get("landing_success"),
                "collision_count": closed.get("collision_count"),
            },
        },
    )


def _supervisor_evidence(root):
    path, value = _latest_valid(
        root.glob("outputs/sandbox/supervisor_gate/*/supervisor_gate.json"),
        inspect_supervisor_gate,
    )
    if path is None:
        missing = {"passed": False, "error": value or "supervisor gate is missing"}
        return missing, missing
    common = {
        "identity": value["supervisor_gate_identity_sha256"],
        "path": path.relative_to(root).as_posix(),
    }
    return (
        {**common, **value["safe_stop"]},
        {**common, **value["failure_diagnostic"]},
    )


def acceptance_snapshot(config):
    root = config.project_root
    replay, closed = _lidar_evidence(config)
    safe_stop, diagnostics = _supervisor_evidence(root)
    checks = {
        "visual_replay": _visual_evidence(root),
        "lidar_replay": replay,
        "closed_loop_flight": closed,
        "safe_process_stop": safe_stop,
        "failure_diagnostics": diagnostics,
    }
    return {"passed": all(value["passed"] for value in checks.values()), "checks": checks}


def _approved_output(project_root, output):
    root = (Path(project_root) / "outputs/sandbox/acceptance").resolve()
    path = Path(output).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("acceptance output is outside the sandbox root") from error
    if path.exists():
        raise ValueError("acceptance output already exists")
    return path


def run_acceptance(project_root, output):
    project_root = Path(project_root).resolve()
    output = _approved_output(project_root, output)
    commit = git_commit(project_root)
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("Sandbox v1 acceptance requires a clean tracked commit")
    run_supervisor_gate(
        project_root, project_root / "outputs/sandbox/supervisor_gate" / output.name
    )
    snapshot = acceptance_snapshot(InspectionConfig.defaults(project_root))
    recipe = {
        "acceptance_recipe_schema_version": ACCEPTANCE_RECIPE_SCHEMA_VERSION,
        "acceptance_profile": "sandbox-v1",
        "software_commit_sha": commit,
        "minimum_visual_frames": MINIMUM_VISUAL_FRAMES,
        "evidence_identities": {
            name: value.get("identity") for name, value in snapshot["checks"].items()
        },
    }
    recipe["acceptance_recipe_identity_sha256"] = object_sha256(recipe)
    result = {
        "acceptance_result_schema_version": ACCEPTANCE_RESULT_SCHEMA_VERSION,
        "acceptance_profile": "sandbox-v1",
        "passed": snapshot["passed"],
        "recipe_identity_sha256": recipe["acceptance_recipe_identity_sha256"],
        "software_commit_sha": commit,
        "checks": snapshot["checks"],
        "scope_note": "Visual and LiDAR evidence is paired; this is not synchronized sensor fusion.",
    }
    result["acceptance_result_identity_sha256"] = object_sha256(result)
    write_json(output / "recipe.json", recipe)
    write_json(output / "acceptance.json", result)
    return result


def inspect_acceptance(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("acceptance_result_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("sandbox acceptance identity mismatch")
    value["acceptance_result_identity_sha256"] = supplied
    return value
