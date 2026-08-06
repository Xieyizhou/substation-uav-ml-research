"""Managed subprocess helpers for local visual collection."""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CollectionProcessError(RuntimeError):
    pass


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_handle: object
    log_path: Path

    def close_log(self):
        if not self.log_handle.closed:
            self.log_handle.close()


def start_process(name, command, log_path, *, env=None, discard_stdout=False):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\nCOMMAND: {' '.join(str(item) for item in command)}\n")
    if discard_stdout:
        handle.write("STDOUT: discarded; STDERR is retained below\n")
    handle.flush()
    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL if discard_stdout else handle,
        stderr=handle if discard_stdout else subprocess.STDOUT,
        start_new_session=True,
    )
    return ManagedProcess(name, process, handle, log_path)


def stop_process(managed, *, grace_s=10.0):
    if managed is None:
        return
    process = managed.process
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired as error:
                raise CollectionProcessError(
                    f"{managed.name} did not stop; inspect {managed.log_path}"
                ) from error
    managed.close_log()


def wait_process(managed, timeout_s):
    try:
        code = managed.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise CollectionProcessError(
            f"{managed.name} exceeded {timeout_s:.0f}s; inspect {managed.log_path}"
        ) from error
    finally:
        managed.close_log()
    if code:
        raise CollectionProcessError(
            f"{managed.name} exited with code {code}; inspect {managed.log_path}"
        )


def wait_for_flight(flight, recorder, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        recorder_code = recorder.process.poll()
        if recorder_code is not None and recorder_code != 0:
            recorder.close_log()
            raise CollectionProcessError(
                f"{recorder.name} exited with code {recorder_code}; "
                f"inspect {recorder.log_path}"
            )
        flight_code = flight.process.poll()
        if flight_code is not None:
            flight.close_log()
            if flight_code:
                raise CollectionProcessError(
                    f"{flight.name} exited with code {flight_code}; "
                    f"inspect {flight.log_path}"
                )
            return
        time.sleep(0.2)
    raise CollectionProcessError(
        f"{flight.name} exceeded {timeout_s:.0f}s; inspect {flight.log_path}"
    )


def ensure_process_running(managed, *, settle_s=1.0):
    time.sleep(settle_s)
    code = managed.process.poll()
    if code is not None:
        managed.close_log()
        raise CollectionProcessError(
            f"{managed.name} exited with code {code}; inspect {managed.log_path}"
        )


def probe_until_ready(
    log_path,
    *,
    startup_timeout_s,
    probe_timeout_s,
    required_process=None,
):
    deadline = time.monotonic() + startup_timeout_s
    command = [
        sys.executable,
        "main.py",
        "visual",
        "gazebo-probe",
        "--timeout",
        str(probe_timeout_s),
    ]
    while time.monotonic() < deadline:
        if required_process is not None:
            ensure_process_running(required_process, settle_s=0.0)
        with Path(log_path).open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=probe_timeout_s + 10.0,
            )
        if result.returncode == 0:
            return
        time.sleep(2.0)
    raise CollectionProcessError(
        f"Gazebo visual topics were not ready within {startup_timeout_s:.0f}s; "
        f"inspect {log_path}"
    )


def wait_for_jsonl_event(path, event_type, required_process, timeout_s):
    path = Path(path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = required_process.process.poll()
        if code is not None:
            required_process.close_log()
            raise CollectionProcessError(
                f"{required_process.name} exited with code {code} before "
                f"{event_type}; inspect {required_process.log_path}"
            )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = ()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == event_type:
                return event
            if event.get("event_type") == "mission_failed":
                raise CollectionProcessError(
                    f"{required_process.name} failed before {event_type}: "
                    f"{event.get('message', 'unknown failure')}"
                )
        time.sleep(0.2)
    raise CollectionProcessError(
        f"{required_process.name} did not publish {event_type} within "
        f"{timeout_s:.0f}s; inspect {required_process.log_path}"
    )
