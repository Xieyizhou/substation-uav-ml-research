"""Identity-bound, preview-first retention for generated sandbox outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from src.ml.artifacts import object_sha256, write_json
from src.sandbox.job_models import SandboxJobStore, TERMINAL_STATES


RETENTION_SCHEMA_VERSION = 1
MINIMUM_AGE_HOURS = 24
DEVELOPMENT_GROUPS = (
    ("operator_jobs", "outputs/sandbox/operator/jobs", 50),
    ("flight_smoke", "outputs/sandbox/flight_smoke", 10),
    ("training_smoke", "outputs/sandbox/training_smoke", 3),
    ("experiments", "outputs/sandbox/experiments", 10),
    ("lidar_replay", "outputs/sandbox/lidar_replay", 10),
    ("supervisor_gate", "outputs/sandbox/supervisor_gate", 5),
    ("acceptance", "outputs/sandbox/acceptance", 5),
)
DEMO_GROUPS = (
    ("demo_jobs", "outputs/sandbox/demo/operator/jobs", 30),
    ("demo_runs", "outputs/sandbox/demo/runs", 5),
    ("demo_release_gate", "outputs/sandbox/demo/release-gate", 3),
)


def _groups(profile):
    if profile == "demo":
        return DEMO_GROUPS
    return DEVELOPMENT_GROUPS if profile == "development" else ()


def _group_root(config, relative):
    declared = config.project_root / relative
    if declared.is_symlink():
        raise ValueError(f"retention group cannot be a symlink: {relative}")
    resolved = declared.resolve()
    sandbox = (config.project_root / "outputs/sandbox").resolve()
    try:
        resolved.relative_to(sandbox)
    except ValueError as error:
        raise ValueError("retention group is outside outputs/sandbox") from error
    return declared


def _tree_record(path, project_root):
    project_root = Path(project_root).resolve()
    path = Path(path).resolve()
    entries, size, newest = [], 0, path.stat().st_mtime_ns
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        stat = candidate.stat()
        relative = candidate.relative_to(project_root).as_posix()
        entries.append((relative, stat.st_size, stat.st_mtime_ns))
        size += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
    return size, newest, object_sha256(entries)


def _terminal_job(path):
    record = path / "job.json"
    if not record.is_file():
        return False
    try:
        return json.loads(record.read_text(encoding="utf-8")).get("state") in TERMINAL_STATES
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _output_path(config, output):
    declared_root = config.project_root / "outputs/sandbox/retention"
    if declared_root.is_symlink():
        raise ValueError("retention plan directory cannot be a symlink")
    root = declared_root.resolve()
    path = Path(output)
    path = path if path.is_absolute() else config.project_root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("retention plans must stay under outputs/sandbox/retention") from error
    return path


def create_retention_plan(config, output, now=None):
    now = now or datetime.now(timezone.utc)
    cutoff_ns = int(now.timestamp() * 1_000_000_000) - (
        MINIMUM_AGE_HOURS * 3600 * 1_000_000_000
    )
    candidates, groups = [], []
    locked = config.profile == "formal"
    for name, relative, keep in _groups(config.profile):
        root = _group_root(config, relative)
        records = []
        for path in root.iterdir() if root.is_dir() else ():
            if not path.is_dir() or path.is_symlink():
                continue
            size, newest, identity = _tree_record(path, config.project_root)
            records.append((newest, path, size, identity))
        records.sort(key=lambda item: (-item[0], item[1].name))
        group_candidates = 0
        for index, (newest, path, size, identity) in enumerate(records):
            terminal = name not in {"operator_jobs", "demo_jobs"} or _terminal_job(path)
            if locked or index < keep or newest >= cutoff_ns or not terminal:
                continue
            candidates.append({
                "path": path.relative_to(config.project_root).as_posix(),
                "bytes": size,
                "tree_identity_sha256": identity,
                "reason": f"{name}: beyond newest {keep} and older than 24 hours",
            })
            group_candidates += 1
        groups.append({
            "name": name, "path": relative, "keep_newest": keep,
            "entry_count": len(records), "candidate_count": group_candidates,
        })
    record = {
        "retention_plan_schema_version": RETENTION_SCHEMA_VERSION,
        "profile": config.profile,
        "created_at": now.isoformat(timespec="seconds"),
        "minimum_age_hours": MINIMUM_AGE_HOURS,
        "formal_retention_locked": locked,
        "groups": groups,
        "candidates": sorted(candidates, key=lambda item: item["path"]),
        "candidate_bytes": sum(item["bytes"] for item in candidates),
    }
    record["retention_plan_identity_sha256"] = object_sha256(record)
    path = _output_path(config, output)
    write_json(path, record)
    return {"path": str(path), "plan": record}


def inspect_retention_plan(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("retention_plan_identity_sha256", None)
    if value.get("retention_plan_schema_version") != RETENTION_SCHEMA_VERSION:
        raise ValueError("unsupported retention plan schema")
    if supplied != object_sha256(value):
        raise ValueError("retention plan identity mismatch")
    value["retention_plan_identity_sha256"] = supplied
    return value


def apply_retention_plan(config, path):
    plan = inspect_retention_plan(path)
    if config.profile == "formal" or plan.get("formal_retention_locked"):
        raise ValueError("formal profile outputs cannot be pruned")
    if plan.get("profile") != config.profile:
        raise ValueError("retention plan profile mismatch")
    if any(job.state not in TERMINAL_STATES for job in SandboxJobStore(
        config.sandbox_jobs_root
    ).list(200)):
        raise ValueError("retention cannot run while a sandbox job is active")
    allowed = {
        _group_root(config, relative).resolve()
        for _, relative, _ in _groups(config.profile)
    }
    verified = []
    for item in plan["candidates"]:
        candidate = (config.project_root / item["path"]).resolve()
        if candidate.parent not in allowed or not candidate.is_dir():
            raise ValueError(f"retention candidate is unavailable: {item['path']}")
        _, _, identity = _tree_record(candidate, config.project_root)
        if identity != item["tree_identity_sha256"]:
            raise ValueError(f"retention candidate changed: {item['path']}")
        verified.append((candidate, item))
    for candidate, _ in verified:
        shutil.rmtree(candidate)
    return {
        "applied": True,
        "retention_plan_identity_sha256": plan["retention_plan_identity_sha256"],
        "removed_count": len(verified),
        "removed_bytes": sum(item["bytes"] for _, item in verified),
        "removed_paths": [item["path"] for _, item in verified],
    }
