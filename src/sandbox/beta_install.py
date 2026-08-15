"""Clean-source installation gate for the dependency-free Demo profile."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.sandbox.gate_outcome import (
    ENVIRONMENT_UNAVAILABLE, PASSED, PRODUCT_FAILURE, validate_outcome,
)


BETA_INSTALL_SCHEMA_VERSION = 2
BETA_INSTALL_VERSION = "1.0.0"


def _tracked_files(root):
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True,
        capture_output=True, timeout=30,
    )
    files = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = PurePosixPath(os.fsdecode(raw))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Git returned an unsafe tracked path")
        source = root.joinpath(*relative.parts)
        if source.is_file():
            files.append((relative.as_posix(), source))
    if not files:
        raise ValueError("no tracked source files were found")
    return tuple(files)


def _copy_source(root, destination):
    files = _tracked_files(root)
    manifest = []
    for relative, source in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append({"path": relative, "sha256": file_sha256(source)})
    return {
        "file_count": len(manifest),
        "source_manifest_sha256": object_sha256(manifest),
    }


def _environment(home, virtual_environment):
    home.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "PATH": os.pathsep.join((str(virtual_environment / "bin"), os.defpath)),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(home),
    }


def _tail(value, limit=2000):
    value = value.strip()
    return value[-limit:] if value else None


def _run_check(name, argv, cwd, environment, timeout):
    started = time.monotonic()
    try:
        result = subprocess.run(
            [str(item) for item in argv], cwd=cwd, env=environment,
            capture_output=True, text=True, timeout=timeout,
        )
        outcome = PASSED if result.returncode == 0 else (
            ENVIRONMENT_UNAVAILABLE
            if name == "app_smoke" and result.returncode == 2
            else PRODUCT_FAILURE
        )
        return {
            "name": name,
            "passed": result.returncode == 0,
            "outcome": outcome,
            "reason_code": (
                None if outcome == PASSED else
                "loopback_bind_denied" if outcome == ENVIRONMENT_UNAVAILABLE else
                "command_failed"
            ),
            "return_code": result.returncode,
            "duration_s": round(time.monotonic() - started, 3),
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "name": name,
            "passed": False,
            "outcome": PRODUCT_FAILURE,
            "reason_code": "command_start_failed",
            "return_code": None,
            "duration_s": round(time.monotonic() - started, 3),
            "stdout_tail": None,
            "stderr_tail": str(error),
        }


def _command(python, *arguments):
    return (python, "main.py", "sandbox", "--profile", "demo", *arguments)


def _workflow_checks(source, python, environment):
    commands = (
        ("bootstrap", _command(python, "bootstrap")),
        ("demo_run", _command(
            python, "demo-run", "--output", "outputs/sandbox/demo/runs/beta",
        )),
        ("demo_inspect", _command(
            python, "demo-inspect", "--input",
            "outputs/sandbox/demo/runs/beta",
        )),
        ("release_gate", _command(
            python, "release-gate", "--output",
            "outputs/sandbox/demo/release-gate/beta",
        )),
        ("release_inspect", _command(
            python, "release-gate-inspect", "--input",
            "outputs/sandbox/demo/release-gate/beta/release_gate.json",
        )),
        ("app_smoke", (python, "-m", "src.sandbox.app_smoke")),
    )
    checks = []
    for name, argv in commands:
        check = _run_check(name, argv, source, environment, 90)
        checks.append(check)
        if not check["passed"]:
            break
    return checks


def run_beta_install_gate(project_root, output, python_executable=None):
    root = Path(project_root).resolve()
    output = Path(output).resolve()
    executable = Path(python_executable or sys.executable).resolve()
    checks = []
    source_identity = None
    with tempfile.TemporaryDirectory(prefix="uav-sandbox-beta-") as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / "source"
        source.mkdir()
        try:
            source_identity = _copy_source(root, source)
            checks.append({
                "name": "source_copy", "passed": True, "outcome": PASSED,
                "reason_code": None, **source_identity,
            })
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            checks.append({
                "name": "source_copy", "passed": False,
                "outcome": PRODUCT_FAILURE, "reason_code": "source_copy_failed",
                "detail": str(error),
            })
        virtual_environment = temporary_root / "venv"
        environment = _environment(temporary_root / "home", virtual_environment)
        if checks[-1]["passed"]:
            checks.append(_run_check(
                "clean_virtual_environment",
                (executable, "-m", "venv", virtual_environment),
                temporary_root, environment, 120,
            ))
        python = virtual_environment / "bin/python"
        if checks[-1]["passed"]:
            checks.extend(_workflow_checks(source, python, environment))
    record = {
        "beta_install_schema_version": BETA_INSTALL_SCHEMA_VERSION,
        "beta_install_version": BETA_INSTALL_VERSION,
        "profile": "demo",
        "source_commit_sha": git_commit(root),
        "source_identity": source_identity,
        "host": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
        "network_required": False,
        "checks": checks,
        "passed": len(checks) == 8 and all(check["passed"] for check in checks),
        "scope_note": (
            "Validates a tracked-source copy, fresh virtual environment, Demo "
            "workflow, release receipt, and loopback App without external assets."
        ),
    }
    record["beta_install_identity_sha256"] = object_sha256(record)
    write_json(output / "beta_install_gate.json", record)
    return record


def inspect_beta_install_gate(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("beta_install_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("beta installation gate identity mismatch")
    if value.get("beta_install_schema_version") != BETA_INSTALL_SCHEMA_VERSION:
        raise ValueError("unsupported beta installation gate schema")
    if value.get("beta_install_version") != BETA_INSTALL_VERSION:
        raise ValueError("unsupported beta installation gate version")
    passed = len(value.get("checks", [])) == 8 and all(
        check.get("passed") is True for check in value.get("checks", [])
    )
    for check in value.get("checks", []):
        validate_outcome(check)
    if value.get("passed") is not passed:
        raise ValueError("beta installation gate status is inconsistent")
    return {**value, "beta_install_identity_sha256": supplied}
