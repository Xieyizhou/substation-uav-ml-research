"""Allow-listed command construction for sandbox operator jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import sys

from src.inspection.dashboard import is_blind, load_plan
from src.sandbox.command_models import SandboxCommand
from src.sandbox.workflow_commands import (
    build_lidar_sandbox_command,
    build_visual_command,
    build_workflow_command,
)
from src.sandbox.profiles import sandbox_profile
from src.sandbox.workbench_commands import build_workbench_command
from src.vision.collection.plan import collection_status
from src.vision.collection.recording import scenario_by_id


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        f"outputs/sandbox/training_smoke/visual-yolo11n-v2-{stamp}",
        "--smoke",
    )


def _doctor_command(config):
    return SandboxCommand(
        "doctor",
        (
            sys.executable, "main.py", "sandbox", "--project-root",
            str(config.project_root), "--profile", config.profile, "doctor",
        ),
        60.0,
        workflow="environment_check",
        requires_runtime_idle=False,
    )


def _flight_command(config, scenario_id):
    row = _scenario(config, scenario_id)
    if is_blind(row):
        raise ValueError("flight smoke cannot use a blind scenario")
    return SandboxCommand(
        "flight-smoke",
        (
            sys.executable, "main.py", "sandbox", "flight-smoke",
            "--plan", str(config.plan_path),
            "--output-root", str(config.collection_root),
            "--scenario-id", str(scenario_id),
        ),
        600.0,
        str(scenario_id),
        workflow="flight_smoke",
        budget_paths=("outputs/sandbox/flight_smoke",),
    )


def _model_command(config, action):
    if action == "training-view-v2":
        return SandboxCommand(
            action, _training_view_command(config), 1_800.0,
            workflow="training_view",
        )
    if action == "training-smoke-v2":
        identity = config.project_root / (
            "data/research/visual_yolo_v2/identity/training_view_identity.json"
        )
        if not identity.is_file():
            raise ValueError("materialize the v2 training view before smoke training")
        return SandboxCommand(
            action, _training_smoke_command(), 3_600.0,
            workflow="training_smoke",
        )
    if action == "package-inspect-v2":
        package = config.project_root / (
            "models/equipment/visual-yolo11n-baseline-v2-package"
        )
        if not (package / "manifest.json").is_file():
            raise ValueError("the frozen v2 model package is not present")
        return SandboxCommand(
            action,
            (
                sys.executable,
                "main.py",
                "visual",
                "model-package-inspect",
                "--input",
                str(package),
            ),
            120.0,
            workflow="package_inspection",
            requires_runtime_idle=False,
        )
    return None


def _collection_job(config, action):
    counts = {"collection-single": 1, "collection-gate": 5}
    if action not in counts:
        return None
    count = counts[action]
    rows = _next_rows(config, count)
    return SandboxCommand(
        action,
        _collection_command(config, count),
        900.0 if count == 1 else 3600.0,
        sensitive=any(is_blind(row) for row in rows),
        workflow="visual_collection",
    )


def build_command(config, action, scenario_id=None, parameters=None):
    profile = sandbox_profile(config.profile)
    if action == "workflow-run":
        return build_workflow_command(config, parameters)
    if action.startswith("workbench-"):
        command = build_workbench_command(config, action, parameters)
        if command is not None:
            return command
    if not profile.flight_enabled and action != "doctor":
        raise ValueError(f"action is unavailable in {config.profile} profile")
    if action == "doctor":
        return _doctor_command(config)
    if action == "flight-smoke":
        return _flight_command(config, scenario_id)
    if action.startswith("lidar-"):
        return build_lidar_sandbox_command(config, action)
    if action == "experiment-run":
        return build_visual_command(config, parameters)
    command = _model_command(config, action) or _collection_job(config, action)
    if command is not None:
        return command
    raise ValueError(f"unsupported sandbox action: {action}")
