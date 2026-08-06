"""Allow-listed command construction for sandbox operator jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from src.inspection.dashboard import is_blind, load_plan
from src.vision.collection.plan import collection_status
from src.vision.collection.recording import scenario_by_id


@dataclass(frozen=True)
class SandboxCommand:
    action: str
    argv: tuple[str, ...]
    timeout_s: float
    scenario_id: str | None = None
    sensitive: bool = False


def _scenario(config, scenario_id):
    if not scenario_id:
        raise ValueError("scenario_id is required")
    return scenario_by_id(load_plan(config), scenario_id)


def _collection_command(config, count):
    return (
        sys.executable,
        "main.py",
        "visual",
        "collection-run",
        "--plan",
        str(config.plan_path),
        "--output-root",
        str(config.collection_root),
        "--max-scenarios",
        str(count),
        "--max-attempts",
        "1",
        "--retry-delay",
        "0",
    )


def _next_rows(config, limit):
    plan = load_plan(config)
    status = collection_status(plan, config.recordings_root)
    if status["next_scenario"] is None:
        return ()
    start = next(
        index
        for index, row in enumerate(plan["scenarios"])
        if row["scenario_id"] == status["next_scenario"]["scenario_id"]
    )
    return tuple(plan["scenarios"][start : start + limit])


def _training_view_command(config):
    return (
        sys.executable,
        "main.py",
        "visual",
        "training-view-materialize",
        "--collection-root",
        str(config.collection_root),
        "--output",
        "data/research/visual_yolo_v2",
    )


def _training_smoke_command():
    return (
        sys.executable,
        "main.py",
        "visual",
        "train-yolo",
        "--config",
        "config/perception/visual_yolo11n_baseline_v2.json",
        "--dataset",
        "data/research/visual_yolo_v2",
        "--output",
        "models/equipment/visual-yolo11n-baseline-v2",
        "--smoke",
    )


def build_command(config, action, scenario_id=None):
    if action == "doctor":
        return SandboxCommand(
            action,
            (sys.executable, "main.py", "sandbox", "doctor"),
            60.0,
        )
    if action == "flight-smoke":
        row = _scenario(config, scenario_id)
        if is_blind(row):
            raise ValueError("flight smoke cannot use a blind scenario")
        return SandboxCommand(
            action,
            (
                sys.executable,
                "main.py",
                "sandbox",
                "flight-smoke",
                "--plan",
                str(config.plan_path),
                "--output-root",
                str(config.collection_root),
                "--scenario-id",
                str(scenario_id),
            ),
            600.0,
            str(scenario_id),
        )
    if action == "training-view-v2":
        return SandboxCommand(action, _training_view_command(config), 1_800.0)
    if action == "training-smoke-v2":
        identity = config.project_root / (
            "data/research/visual_yolo_v2/identity/training_view_identity.json"
        )
        if not identity.is_file():
            raise ValueError("materialize the v2 training view before smoke training")
        return SandboxCommand(action, _training_smoke_command(), 3_600.0)
    counts = {"collection-single": 1, "collection-gate": 5}
    if action in counts:
        count = counts[action]
        rows = _next_rows(config, count)
        sensitive = any(is_blind(row) for row in rows)
        return SandboxCommand(
            action,
            _collection_command(config, count),
            900.0 if count == 1 else 3600.0,
            sensitive=sensitive,
        )
    raise ValueError(f"unsupported sandbox action: {action}")
