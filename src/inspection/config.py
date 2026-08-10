"""Configuration and centralized allow-listed path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class AccessDenied(ValueError):
    """Requested content is outside the configured inspection boundary."""


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise AccessDenied("path is outside the approved root") from error
    return candidate


@dataclass(frozen=True)
class InspectionConfig:
    project_root: Path
    plan_path: Path
    collection_root: Path
    px4_root: Path
    minimum_free_gib: float = 5.0

    @classmethod
    def defaults(cls, project_root: Path) -> "InspectionConfig":
        root = Path(project_root).resolve()
        collection = root / "data/research/visual_collection_v2"
        local_plan = collection / "collection_plan.json"
        plan = local_plan if local_plan.is_file() else (
            root / "benchmarks/visual_static_v2/collection_plan.json"
        )
        return cls(root, plan, collection, Path.home() / "PX4-Autopilot")

    @property
    def recordings_root(self) -> Path:
        return self.collection_root / "recordings"

    @property
    def sandbox_operator_root(self) -> Path:
        return self.project_root / "outputs/sandbox/operator"

    @property
    def sandbox_jobs_root(self) -> Path:
        return self.sandbox_operator_root / "jobs"

    @property
    def sandbox_experiments_root(self) -> Path:
        return self.project_root / "outputs/sandbox/experiments"

    def recording(self, recording_id: str) -> Path:
        if not recording_id or Path(recording_id).name != recording_id:
            raise AccessDenied("invalid recording identifier")
        return _inside(self.recordings_root, self.recordings_root / recording_id)

    def frame_payload(self, recording_id: str, relative: str) -> Path:
        root = self.recording(recording_id)
        candidate = _inside(root, root / relative)
        if candidate.suffix.lower() != ".png":
            raise AccessDenied("only stored PNG frames may be viewed")
        return candidate

    def log(self, scenario_id: str, kind: str) -> Path:
        names = {name: f"{name}.log" for name in (
            "recorder", "flight", "simulator", "probe"
        )}
        if kind not in names or Path(scenario_id).name != scenario_id:
            raise AccessDenied("invalid log selection")
        root = self.collection_root / "batch_logs"
        return _inside(root, root / scenario_id / names[kind])
