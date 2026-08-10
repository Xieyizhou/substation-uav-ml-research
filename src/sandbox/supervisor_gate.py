"""Offline acceptance checks for sandbox process supervision."""

from __future__ import annotations

from pathlib import Path
import sys

from src.ml.artifacts import object_sha256, write_json
from src.sandbox.job_process import process_alive, start_job_process, stop_job_process


SUPERVISOR_GATE_SCHEMA_VERSION = 1
FAILURE_SENTINEL = "sandbox supervisor diagnostic sentinel"


def _approved_output(project_root, output):
    root = (Path(project_root) / "outputs/sandbox/supervisor_gate").resolve()
    path = Path(output).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("supervisor gate output is outside the sandbox root") from error
    if (path / "supervisor_gate.json").exists():
        raise ValueError("supervisor gate output already exists")
    return path


def _safe_stop(project_root, output):
    process, handle = start_job_process(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        output / "safe_stop.log",
        project_root,
    )
    try:
        exit_code = stop_job_process(
            process, interrupt_s=1.0, terminate_s=1.0
        )
    finally:
        handle.close()
    return {
        "passed": exit_code is not None and not process_alive(process.pid),
        "exit_code": exit_code,
        "process_alive_after_stop": process_alive(process.pid),
    }


def _failure_diagnostic(project_root, output):
    script = (
        "import sys; "
        f"print({FAILURE_SENTINEL!r}, file=sys.stderr); "
        "raise SystemExit(7)"
    )
    process, handle = start_job_process(
        (sys.executable, "-c", script),
        output / "failure.log",
        project_root,
    )
    try:
        exit_code = process.wait(timeout=5.0)
    finally:
        handle.close()
    log = (output / "failure.log").read_text(encoding="utf-8", errors="replace")
    return {
        "passed": exit_code == 7 and FAILURE_SENTINEL in log,
        "exit_code": exit_code,
        "diagnostic_detected": FAILURE_SENTINEL in log,
    }


def run_supervisor_gate(project_root, output):
    output = _approved_output(project_root, output)
    output.mkdir(parents=True, exist_ok=True)
    safe_stop = _safe_stop(project_root, output)
    failure = _failure_diagnostic(project_root, output)
    report = {
        "supervisor_gate_schema_version": SUPERVISOR_GATE_SCHEMA_VERSION,
        "passed": safe_stop["passed"] and failure["passed"],
        "safe_stop": safe_stop,
        "failure_diagnostic": failure,
    }
    report["supervisor_gate_identity_sha256"] = object_sha256(report)
    write_json(output / "supervisor_gate.json", report)
    return report


def inspect_supervisor_gate(path):
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("supervisor_gate_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("supervisor gate identity mismatch")
    value["supervisor_gate_identity_sha256"] = supplied
    return value
