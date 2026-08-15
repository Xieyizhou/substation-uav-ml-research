"""Narrow, mockable process inspection adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Protocol

from src.inspection.models import RuntimeItem


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    executable: str
    argv: tuple[str, ...]


class ProcessAdapter(Protocol):
    def processes(self) -> tuple[ProcessRecord, ...]: ...


class ProcessInspectionUnavailable(RuntimeError):
    """Raised when the host does not permit read-only process inspection."""


class LocalProcessAdapter:
    def processes(self) -> tuple[ProcessRecord, ...]:
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,comm=,args="],
                check=False, capture_output=True,
                text=True, timeout=3,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProcessInspectionUnavailable from exc
        if result.returncode != 0:
            raise ProcessInspectionUnavailable
        records = []
        for line in result.stdout.splitlines():
            fields = line.strip().split(maxsplit=3)
            if len(fields) != 4 or not all(item.isdigit() for item in fields[:2]):
                continue
            try:
                argv = tuple(shlex.split(fields[3]))
            except ValueError:
                argv = ()
            records.append(ProcessRecord(
                pid=int(fields[0]), ppid=int(fields[1]),
                executable=fields[2], argv=argv,
            ))
        return tuple(records)


PROCESS_NAMES = (
    "collection runner", "PX4", "Gazebo", "recorder", "flight task",
)
SHELL_EXECUTABLES = frozenset({"bash", "dash", "fish", "sh", "zsh"})


def _basename(value: str) -> str:
    return Path(value).name.lower()


def _has_sequence(argv: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    lowered = tuple(item.lower() for item in argv)
    width = len(sequence)
    return any(lowered[index:index + width] == sequence
               for index in range(len(lowered) - width + 1))


def _process_role(process: ProcessRecord) -> str | None:
    executable = _basename(process.executable)
    argv0 = _basename(process.argv[0]) if process.argv else executable
    if executable in SHELL_EXECUTABLES or argv0 in SHELL_EXECUTABLES:
        return None
    if executable == "px4" or argv0 == "px4":
        return "PX4"
    if executable in {"gzserver", "gazebo"} or argv0 in {"gzserver", "gazebo"}:
        return "Gazebo"
    if (executable == "gz" or argv0 == "gz") and "sim" in {
        item.lower() for item in process.argv[1:]
    }:
        return "Gazebo"
    if _has_sequence(process.argv, ("visual", "collection-run")):
        return "collection runner"
    if any(_has_sequence(process.argv, ("visual", command)) for command in (
        "collection-record", "pilot-record", "visual-pilot-record",
    )):
        return "recorder"
    script_names = {_basename(item) for item in process.argv}
    if script_names.intersection({"run_task.py", "fly_astar.py", "fly_astar_path.py"}):
        return "flight task"
    return None


def runtime_status(adapter: ProcessAdapter) -> tuple[RuntimeItem, ...]:
    try:
        processes = adapter.processes()
    except ProcessInspectionUnavailable:
        return tuple(RuntimeItem(
            name=name, available=False, alive=False, pid=None,
            detail="process inspection unavailable; no restart attempted",
        ) for name in PROCESS_NAMES)
    matches = {}
    for process in processes:
        role = _process_role(process)
        if role is not None and role not in matches:
            matches[role] = process
    views = []
    for name in PROCESS_NAMES:
        match = matches.get(name)
        views.append(RuntimeItem(
            name=name, available=True, alive=match is not None,
            pid=None if match is None else match.pid,
            detail="appears alive" if match else "not detected; no restart attempted",
        ))
    return tuple(views)
