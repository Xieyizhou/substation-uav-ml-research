"""Narrow, mockable process inspection adapter."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Protocol

from src.inspection.models import RuntimeItem


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    command: str


class ProcessAdapter(Protocol):
    def processes(self) -> tuple[ProcessRecord, ...]: ...


class ProcessInspectionUnavailable(RuntimeError):
    """Raised when the host does not permit read-only process inspection."""


class LocalProcessAdapter:
    def processes(self) -> tuple[ProcessRecord, ...]:
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,args="], check=False, capture_output=True,
                text=True, timeout=3,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProcessInspectionUnavailable from exc
        if result.returncode != 0:
            raise ProcessInspectionUnavailable
        records = []
        for line in result.stdout.splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2 and fields[0].isdigit():
                records.append(ProcessRecord(int(fields[0]), fields[1]))
        return tuple(records)


PROCESS_MARKERS = {
    "collection runner": ("collection-run",),
    "PX4": ("px4",),
    "Gazebo": ("gz sim", "gzserver", "gazebo"),
    "recorder": ("collection-record", "visual-pilot-record"),
    "flight task": ("run_task.py", "fly_astar"),
}


def runtime_status(adapter: ProcessAdapter) -> tuple[RuntimeItem, ...]:
    try:
        processes = adapter.processes()
    except ProcessInspectionUnavailable:
        return tuple(RuntimeItem(
            name=name, available=False, alive=False, pid=None,
            detail="process inspection unavailable; no restart attempted",
        ) for name in PROCESS_MARKERS)
    views = []
    for name, markers in PROCESS_MARKERS.items():
        match = next((item for item in processes if any(
            marker.lower() in item.command.lower() for marker in markers
        )), None)
        views.append(RuntimeItem(
            name=name, available=True, alive=match is not None,
            pid=None if match is None else match.pid,
            detail="appears alive" if match else "not detected; no restart attempted",
        ))
    return tuple(views)
