"""Small PID guard shared by the compact live-flight task programs."""

import os
from contextlib import contextmanager

from src.maps.target_catalog import FLIGHT_PID_PATH, flight_runner_pid


def _remove_own_pid():
    try:
        recorded_pid = int(FLIGHT_PID_PATH.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return
    if recorded_pid == os.getpid():
        FLIGHT_PID_PATH.unlink(missing_ok=True)


@contextmanager
def managed_task_runtime(task_args):
    """Track compact live tasks while leaving dry-run previews untracked."""
    if "--dry-run" in task_args:
        yield
        return

    active_pid = flight_runner_pid()
    if active_pid not in {None, os.getpid()}:
        raise RuntimeError(
            f"Another project-managed flight is already running (PID {active_pid})"
        )

    FLIGHT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = FLIGHT_PID_PATH.with_suffix(".pid.tmp")
    temporary_path.write_text(f"{os.getpid()}\n")
    temporary_path.replace(FLIGHT_PID_PATH)
    try:
        yield
    finally:
        _remove_own_pid()
