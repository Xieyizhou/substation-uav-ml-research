"""Idempotent local initialization for sandbox runtime directories."""

from __future__ import annotations

import json
from pathlib import Path
import platform

from src.ml.artifacts import git_commit, object_sha256, write_json
from src.sandbox.profiles import sandbox_profile


BOOTSTRAP_SCHEMA_VERSION = 1


def _validate_demo_plan(config):
    value = json.loads(config.plan_path.read_text(encoding="utf-8"))
    supplied = value.pop("collection_plan_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("demo collection plan identity mismatch")
    if value.get("profile") != "demo" or value.get("scenarios") != []:
        raise ValueError("demo collection plan must contain no formal scenarios")
    return supplied


def bootstrap_sandbox(config, output: Path | None = None):
    profile = sandbox_profile(config.profile)
    if not config.project_root.is_dir():
        raise FileNotFoundError(config.project_root)
    plan_identity = (
        _validate_demo_plan(config)
        if profile.profile_id == "demo"
        else json.loads(config.plan_path.read_text(encoding="utf-8"))[
            "collection_plan_identity_sha256"
        ]
    )
    for directory in (
        config.collection_root,
        config.recordings_root,
        config.sandbox_operator_root,
        config.sandbox_jobs_root,
        config.sandbox_experiments_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    worlds = sorted(config.project_root.glob("simulation/worlds/*.sdf"))
    if not worlds:
        raise FileNotFoundError("tracked Gazebo world definitions are missing")
    record = {
        "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "profile": profile.to_record(),
        "python_version": platform.python_version(),
        "software_commit_sha": git_commit(config.project_root),
        "collection_plan_identity_sha256": plan_identity,
        "runtime_directories": [
            path.relative_to(config.project_root).as_posix()
            for path in (
                config.collection_root,
                config.sandbox_operator_root,
                config.sandbox_experiments_root,
            )
        ],
        "world_definition_count": len(worlds),
        "ready": True,
    }
    record["bootstrap_identity_sha256"] = object_sha256(record)
    destination = Path(output or config.sandbox_bootstrap_root / "bootstrap.json")
    write_json(destination, record)
    return {**record, "path": str(destination)}


def inspect_bootstrap(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("bootstrap_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("sandbox bootstrap identity mismatch")
    if value.get("ready") is not True:
        raise ValueError("sandbox bootstrap is not ready")
    sandbox_profile(value["profile"]["profile_id"])
    return {**value, "bootstrap_identity_sha256": supplied}
