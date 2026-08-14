"""Read-only preflight status and capability-receipt enforcement."""

from __future__ import annotations

from pathlib import Path

from src.inspection.doctor import run_doctor
from src.inspection.runtime import LocalProcessAdapter, runtime_status
from src.study.challenge_receipt import inspect_challenge_receipt


def _registry(root):
    return Path(root) / "outputs/research/registry.sqlite"


def _receipt_paths(root):
    return sorted(
        Path(root).glob(
            "outputs/research/study_results/*/challenge/challenge_receipt.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def latest_challenge_receipt(root, model_id=None):
    error = None
    for path in _receipt_paths(root):
        try:
            value = inspect_challenge_receipt(
                path, project_root=root, registry_path=_registry(root)
            )
        except (KeyError, OSError, TypeError, ValueError) as current:
            error = str(current)
            continue
        if model_id is None or value.get("model_id") == model_id:
            status = "passed" if value["passed"] and value["current"] else (
                "stale" if value["passed"] else "failed"
            )
            return {**value, "status": status}
    return {"status": "invalid" if error else "missing", "passed": False,
            "current": False, "error": error}


def require_current_challenge(root, model_id):
    value = latest_challenge_receipt(root, model_id)
    if not value.get("passed") or not value.get("current"):
        detail = value.get("error") or value.get("status", "missing")
        raise ValueError(
            "a passed capability challenge for the current clean commit and model "
            f"is required before the large LiDAR gate ({detail})"
        )
    return value


def preflight_summary(config, process_adapter=None):
    doctor = run_doctor(config)
    runtime = runtime_status(process_adapter or LocalProcessAdapter())
    challenge = latest_challenge_receipt(config.project_root)
    doctor_ready = not any(item.status == "failure" for item in doctor)
    disk = next((item for item in doctor if item.name == "Disk space"), None)
    runtime_idle = not any(item.alive for item in runtime)
    checks = {
        "doctor": {
            "passed": doctor_ready,
            "detail": "required environment checks passed" if doctor_ready
            else "one or more required environment checks failed",
        },
        "disk": {
            "passed": bool(disk and disk.status == "pass"),
            "detail": disk.explanation if disk else "disk check unavailable",
        },
        "runtime_idle": {
            "passed": runtime_idle,
            "detail": "no managed simulator process detected" if runtime_idle
            else "a simulator or flight process is already active",
        },
        "capability_challenge": {
            "passed": bool(challenge.get("passed") and challenge.get("current")),
            "detail": challenge.get("status", "missing"),
        },
    }
    return {
        "status": "ready" if all(item["passed"] for item in checks.values())
        else "blocked",
        "ready_for_large_lidar": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "challenge": challenge,
    }
