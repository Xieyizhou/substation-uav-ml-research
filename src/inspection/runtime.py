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


class LocalProcessAdapter:
    def processes(self) -> tuple[ProcessRecord, ...]:
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,args="], check=False, capture_output=True,
                text=True, timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
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
    processes = adapter.processes()
    views = []
    for name, markers in PROCESS_MARKERS.items():
        match = next((item for item in processes if any(
            marker.lower() in item.command.lower() for marker in markers
        )), None)
        views.append(RuntimeItem(
            name, match is not None, None if match is None else match.pid,
            "appears alive" if match else "not detected; no restart attempted",
        ))
    return tuple(views)
