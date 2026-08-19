"""Configuration and centralized allow-listed path resolution."""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    profile: str = "development"

    @classmethod
    def defaults(cls, project_root: Path) -> "InspectionConfig":
        return cls.for_profile(project_root, "development")

    @classmethod
    def for_profile(cls, project_root: Path, profile: str) -> "InspectionConfig":
        from src.sandbox.profiles import sandbox_profile

        root = Path(project_root).resolve()
        selected = sandbox_profile(profile)
        px4_root = Path(os.environ.get("PX4_ROOT", Path.home() / "PX4-Autopilot"))
        if selected.profile_id == "demo":
            return cls(
                root,
                root / "config/sandbox/demo_collection_plan.json",
                root / "outputs/sandbox/demo/collection",
                px4_root,
                profile=selected.profile_id,
            )
        collection = root / "data/research/visual_collection_v2"
        local_plan = collection / "collection_plan.json"
        plan = local_plan if local_plan.is_file() else (
            root / "benchmarks/visual_static_v2/collection_plan.json"
        )
        return cls(
            root, plan, collection, px4_root,
            profile=selected.profile_id,
        )

    @property
    def recordings_root(self) -> Path:
        return self.collection_root / "recordings"

    @property
    def sandbox_operator_root(self) -> Path:
        suffix = "operator" if self.profile == "development" else f"{self.profile}/operator"
        return self.project_root / "outputs/sandbox" / suffix

    @property
    def sandbox_jobs_root(self) -> Path:
        return self.sandbox_operator_root / "jobs"

    @property
    def sandbox_experiments_root(self) -> Path:
        suffix = "experiments" if self.profile == "development" else f"{self.profile}/experiments"
        return self.project_root / "outputs/sandbox" / suffix

    @property
    def workbench_root(self) -> Path:
        suffix = "workbench" if self.profile == "development" else f"{self.profile}/workbench"
        return self.project_root / "outputs/sandbox" / suffix

    @property
    def workbench_datasets_root(self) -> Path:
        return self.workbench_root / "datasets"

    @property
    def workbench_runs_root(self) -> Path:
        return self.workbench_root / "runs"

    @property
    def workbench_inbox_root(self) -> Path:
        return self.workbench_root / "inbox"

    @property
    def workbench_inference_root(self) -> Path:
        return self.workbench_root / "inference"

    def workbench_inbox_file(self, name: str) -> Path:
        if not name or Path(name).name != name:
            raise AccessDenied("invalid workbench inbox filename")
        candidate = _inside(self.workbench_inbox_root, self.workbench_inbox_root / name)
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise AccessDenied("workbench inference accepts PNG or JPEG")
        return candidate

    def workbench_inference(self, inference_id: str) -> Path:
        if not inference_id or Path(inference_id).name != inference_id:
            raise AccessDenied("invalid workbench inference identifier")
        return _inside(
            self.workbench_inference_root,
            self.workbench_inference_root / inference_id,
        )

    def workbench_inference_image(self, inference_id: str, name: str) -> Path:
        if name not in {"input.png", "primary.png", "comparison.png"}:
            raise AccessDenied("invalid workbench inference image")
        root = self.workbench_inference(inference_id)
        return _inside(root, root / name)

    @property
    def sandbox_bootstrap_root(self) -> Path:
        return self.project_root / "outputs/sandbox" / self.profile / "bootstrap"

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
