"""Allow-listed command construction for sandbox operator jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
import json

from src.inspection.dashboard import is_blind, load_plan
from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_store import SandboxMapStore
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


def _display_mode(parameters):
    values = dict(parameters or {})
    unknown = set(values) - {"display_mode"}
    if unknown:
        raise ValueError("unsupported simulator parameter: " + sorted(unknown)[0])
    mode = values.get("display_mode", "headless")
    if mode not in {"headless", "visual_preview"}:
        raise ValueError(f"unsupported simulator display mode: {mode}")
    return mode


def _collection_command(config, count, display_mode):
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
        "--display-mode",
        display_mode,
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


def _flight_command(config, scenario_id, display_mode):
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
            "--display-mode", display_mode,
        ),
        600.0,
        str(scenario_id),
        workflow="flight_smoke",
        budget_paths=("outputs/sandbox/flight_smoke",),
    )


def _map_flight_command(config, parameters, *, record=False):
    values = dict(parameters or {})
    allowed = {"map_id", "revision_id", "mission_id", "display_mode"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError("unsupported map flight parameter: " + sorted(unknown)[0])
    required = ("map_id", "revision_id", "mission_id")
    if any(not values.get(name) for name in required):
        raise ValueError("map flight requires map_id, revision_id, and mission_id")
    mode = values.get("display_mode", "headless")
    if mode not in {"headless", "visual_preview"}:
        raise ValueError(f"unsupported simulator display mode: {mode}")
    store = SandboxMapStore(config.sandbox_maps_root)
    _, map_record, _, _ = store.accepted_route_artifacts(
        values["map_id"], values["revision_id"], values["mission_id"]
    )
    map_value = SandboxMap.from_record(map_record)
    if values["mission_id"] not in {item.mission_id for item in map_value.missions}:
        raise ValueError("sandbox map mission was not found")
    argv = (
            sys.executable, "main.py", "sandbox", "--project-root",
            str(config.project_root), "--profile", config.profile, "map-run",
            "--map-id", str(values["map_id"]),
            "--revision-id", str(values["revision_id"]),
            "--mission-id", str(values["mission_id"]),
            "--display-mode", mode,
        )
    if record:
        argv = (*argv, "--record")
    action = "map-record" if record else "map-flight-smoke"
    return SandboxCommand(
        action,
        argv,
        900.0,
        workflow="sandbox_map_record" if record else "sandbox_map_flight",
        budget_paths=("outputs/sandbox/map_runs",),
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


def _collection_job(config, action, parameters):
    counts = {"collection-single": 1, "collection-gate": 5}
    if action not in counts:
        return None
    count = counts[action]
    display_mode = _display_mode(parameters)
    rows = _next_rows(config, count)
    return SandboxCommand(
        action,
        _collection_command(config, count, display_mode),
        900.0 if count == 1 else 3600.0,
        sensitive=any(is_blind(row) for row in rows),
        workflow="visual_collection",
    )


def build_command(config, action, scenario_id=None, parameters=None):
    profile = sandbox_profile(config.profile)
    if action == "evidence-verify":
        if scenario_id is not None or parameters:
            raise ValueError("evidence verification accepts no overrides")
        from uuid import uuid4
        from src.sandbox.evidence_registry import registered_archive, CHECKS, REGISTRY
        from src.sandbox.workflow import WorkflowArtifact
        from src.ml.artifacts import file_sha256
        _, archive = registered_archive(config.project_root)
        if not archive.is_file():
            raise ValueError("Historical evidence archive is not installed")
        output = (CHECKS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex + ".json")).as_posix()
        return SandboxCommand(action, (sys.executable, "main.py", "sandbox", "--project-root",
            str(config.project_root), "evidence-flight-verify", "--output", output), 300.,
            workflow="historical_evidence_verification",
            artifacts=(WorkflowArtifact("archive_registration", file_sha256(config.project_root / REGISTRY), REGISTRY.as_posix()),),
            expected_outputs=(output,), budget_paths=(output,), requires_runtime_idle=False)
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
    if action == "semantic-flight":
        from src.sandbox.semantic_commands import build_semantic_command
        return build_semantic_command(config, scenario_id, parameters)
    if action == "visual-replan-flight":
        if config.profile != "development" or scenario_id is not None or parameters:
            raise ValueError("visual replan is a fixed development-only SITL scenario; no overrides")
        from uuid import uuid4
        from src.sandbox.visual_replan_gate import require_visual_gate, BASE, POSITIVE_RUN, STOP_PROOF
        from src.sandbox.workflow import WorkflowArtifact
        try:
            gate = require_visual_gate(config.project_root)
        except (OSError, KeyError, TypeError) as error:
            raise ValueError(f"visual-replan gate unavailable or incomplete: {error}") from error
        run_id = "sandbox-replan-v1-" + uuid4().hex
        output = (BASE / run_id).as_posix()
        return SandboxCommand(action,
            (sys.executable, "-m", "src.sandbox.visual_replan_entry", "--fly", "--run-id", run_id),
            600., scenario_id=run_id, workflow="bounded_visual_standoff_sitl",
            artifacts=(WorkflowArtifact("visual_gate", gate["positive_receipt_sha256"], (BASE / POSITIVE_RUN / "completion.json").as_posix()),
                       WorkflowArtifact("visual_stop_gate", gate["stop_receipt_sha256"], STOP_PROOF.as_posix())),
            expected_outputs=(output+"/completion.json", output+"/runtime/receipt.json", output+"/vision/receipt.json"),
            budget_paths=(output,))
    if action == "live-replan-flight":
        if config.profile != "development" or scenario_id is not None or parameters:
            raise ValueError("live replan is a fixed development-only SITL scenario; no overrides")
        from uuid import uuid4
        from src.sandbox.live_replan_gate import require_repeat_gate, BASE, CAMPAIGN
        from src.sandbox.workflow import WorkflowArtifact
        try:
            gate = require_repeat_gate(config.project_root)
        except (OSError, KeyError, TypeError) as error:
            raise ValueError(f"live-replan gate unavailable or incomplete: {error}") from error
        run_id = "sandbox-replan-v1-" + uuid4().hex
        output = (BASE / run_id).as_posix()
        return SandboxCommand(
            action,
            (sys.executable, "-m", "scripts.flight.fly_sandbox_replan", "--fly", "--run-id", run_id),
            600., scenario_id=run_id, workflow="bounded_live_replan_sitl",
            artifacts=(WorkflowArtifact("repeat_gate", gate["campaign_hash"], (BASE / CAMPAIGN / "completion.json").as_posix()),),
            expected_outputs=(output + "/completion.json", output + "/runtime/receipt.json"),
            budget_paths=(output,),
        )
    if action == "flight-smoke":
        return _flight_command(config, scenario_id, _display_mode(parameters))
    if action in {"map-flight-smoke", "map-record"}:
        return _map_flight_command(config, parameters, record=action == "map-record")
    if action.startswith("lidar-"):
        return build_lidar_sandbox_command(config, action)
    if action == "experiment-run":
        return build_visual_command(config, parameters)
    command = _model_command(config, action) or _collection_job(
        config, action, parameters
    )
    if command is not None:
        return command
    raise ValueError(f"unsupported sandbox action: {action}")
