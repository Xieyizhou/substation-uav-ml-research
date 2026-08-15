"""Bounded process-group lifecycle for sandbox jobs."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def start_job_process(
    argv, log_path, project_root, result_path=None,
    ownership_path=None, ownership_token=None,
):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    handle.write(f"\nACTION START: {argv[2] if len(argv) > 2 else argv[0]}\n")
    handle.flush()
    managed = result_path is not None
    if managed and (ownership_path is None or not ownership_token):
        raise ValueError("managed process ownership is required")
    command = tuple(argv) if not managed else (
        sys.executable, str(Path(__file__).with_name("process_runner.py")),
        "--result", str(result_path),
        "--ownership", str(ownership_path),
        "--token", ownership_token,
        "--", *argv,
    )
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def ownership_is_held(path, expected_token):
    if not expected_token:
        return False
    try:
        handle = Path(path).open("r+", encoding="utf-8")
        if handle.readline().strip() != expected_token:
            handle.close()
            return False
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
    except OSError:
        return False
    return False


def _snapshot_process_groups(root_pid):
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return (root_pid,)
    children = {}
    groups = {root_pid}
    for line in result.stdout.splitlines():
        try:
            pid, parent, group = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        children.setdefault(parent, []).append((pid, group))
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for pid, group in children.get(parent, ()):
            groups.add(group)
            pending.append(pid)
    return tuple(sorted(groups, reverse=True))


def _signal_groups(groups, number):
    for group in groups:
        try:
            os.killpg(group, number)
        except (ProcessLookupError, PermissionError):
            continue


def _wait(process, timeout_s):
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None


def stop_job_process(process, *, interrupt_s=20.0, terminate_s=10.0):
    groups = _snapshot_process_groups(process.pid)
    _signal_groups(groups, signal.SIGINT)
    code = _wait(process, interrupt_s)
    if code is not None:
        return code
    _signal_groups(groups, signal.SIGTERM)
    code = _wait(process, terminate_s)
    if code is not None:
        return code
    _signal_groups(groups, signal.SIGKILL)
    return _wait(process, 5.0)


def stop_job_pid(
    pid, *, interrupt_s=20.0, terminate_s=10.0, stopped=None,
):
    groups = _snapshot_process_groups(int(pid))
    has_stopped = stopped or (lambda: not process_alive(pid))
    for number, timeout in (
        (signal.SIGINT, interrupt_s), (signal.SIGTERM, terminate_s),
        (signal.SIGKILL, 5.0),
    ):
        _signal_groups(groups, number)
        deadline = time.monotonic() + timeout
        while not has_stopped() and time.monotonic() < deadline:
            time.sleep(0.1)
        if has_stopped():
            return True
    return False


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, ValueError):
        return False
    return True
