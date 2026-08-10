"""Offline release gate for the dependency-free Sandbox v0.1 profile."""

from __future__ import annotations

import json
from pathlib import Path

from src.inspection.doctor import run_doctor
from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.sandbox.bootstrap import bootstrap_sandbox, inspect_bootstrap
from src.sandbox.demo_workflow import inspect_demo, run_demo


RELEASE_VERSION = "0.1.0"
RELEASE_GATE_SCHEMA_VERSION = 1


def _check(passed, identity=None, detail=None):
    return {"passed": bool(passed), "identity": identity, "detail": detail}


def run_release_gate(config, output):
    if config.profile != "demo":
        raise ValueError("Sandbox v0.1 release gate requires the demo profile")
    root = Path(output)
    bootstrap = bootstrap_sandbox(config, root / "bootstrap.json")
    demo = run_demo(config.project_root, root / "demo")
    verified_bootstrap = inspect_bootstrap(root / "bootstrap.json")
    verified_demo = inspect_demo(root / "demo")
    doctor = run_doctor(config)
    required_assets = tuple(
        config.project_root / "src/inspection/static" / name
        for name in ("index.html", "app.js", "style.css", "profile.css")
    )
    asset_identity = object_sha256({
        path.name: file_sha256(path) for path in required_assets if path.is_file()
    })
    checks = {
        "bootstrap": _check(
            verified_bootstrap["ready"], bootstrap["bootstrap_identity_sha256"]
        ),
        "demo_workflow": _check(
            verified_demo["passed"], verified_demo["result_identity_sha256"]
        ),
        "doctor": _check(
            not any(item.status == "failure" for item in doctor),
            object_sha256([item.to_dict() for item in doctor]),
            "Warnings identify optional full-simulator capabilities.",
        ),
        "web_assets": _check(
            all(path.is_file() and path.stat().st_size > 0 for path in required_assets),
            asset_identity,
        ),
    }
    record = {
        "release_gate_schema_version": RELEASE_GATE_SCHEMA_VERSION,
        "release_version": RELEASE_VERSION,
        "profile": config.profile,
        "software_commit_sha": git_commit(config.project_root),
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "scope_note": (
            "Validates local startup, deterministic demo execution, and static App "
            "assets without requiring datasets, model weights, PX4, or Gazebo."
        ),
    }
    record["release_gate_identity_sha256"] = object_sha256(record)
    write_json(root / "release_gate.json", record)
    return record


def inspect_release_gate(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("release_gate_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("sandbox release gate identity mismatch")
    if value.get("release_version") != RELEASE_VERSION:
        raise ValueError("unsupported sandbox release version")
    passed = all(item.get("passed") is True for item in value.get("checks", {}).values())
    if value.get("passed") is not passed:
        raise ValueError("sandbox release gate status is inconsistent")
    return {**value, "release_gate_identity_sha256": supplied}
