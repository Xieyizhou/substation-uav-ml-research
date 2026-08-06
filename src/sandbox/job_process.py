"""Bounded process-group lifecycle for sandbox jobs."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


def start_job_process(argv, log_path, project_root):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    handle.write(f"\nACTION START: {argv[2] if len(argv) > 2 else argv[0]}\n")
    handle.flush()
    process = subprocess.Popen(
        argv,
        cwd=project_root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


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


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, ValueError):
        return False
    return True
