"""Typed read models returned by inspection services."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ReadModel:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorCheck(ReadModel):
    name: str
    status: str
    explanation: str
    action: str = ""


@dataclass(frozen=True)
class ScenarioView(ReadModel):
    scenario_id: str
    recording_id: str
    state: str
    dataset_role: str
    split: str
    layout: str | None
    route: str | None
    seed: int | None
    target_class: str | None


@dataclass(frozen=True)
class DashboardView(ReadModel):
    counts: dict[str, int]
    role_counts: dict[str, int]
    completed: int
    total: int
    remaining: int
    progress_percent: float
    current: ScenarioView | None
    next: ScenarioView | None


@dataclass(frozen=True)
class RuntimeItem(ReadModel):
    name: str
    alive: bool
    pid: int | None
    detail: str


@dataclass(frozen=True)
class LogLine(ReadModel):
    number: int
    text: str
    level: str


@dataclass(frozen=True)
class FrameView(ReadModel):
    frame_id: str
    sequence: int
    simulation_timestamp: float
    mission_phase: str
    synchronization_status: str
    annotation_status: str
    annotation_summary: str
    width: int
    height: int
    image_url: str
    boxes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FramePage(ReadModel):
    recording_id: str
    page: int
    page_size: int
    total_frames: int
    frames: tuple[FrameView, ...]


@dataclass(frozen=True)
class ProgressView(ReadModel):
    recording_id: str
    frame_count: int
    elapsed_simulation_time: float | None
    target_class: str | None
    validation_receipt: str
    phases: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    truth_counts: dict[str, int]
