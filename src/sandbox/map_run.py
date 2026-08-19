"""Execute one validated sandbox-map revision through PX4 and Gazebo."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_store import SandboxMapStore
from src.vision.collection.display import launcher_environment
from src.vision.collection.process import (
    CollectionProcessError, ensure_process_running, probe_until_ready,
    start_process, stop_process, wait_process,
)
from src.vision.collection.world import materialize_camera_model


def _mission(map_value, mission_id):
    try:
        return next(item for item in map_value.missions if item.mission_id == mission_id)
    except StopIteration as error:
        raise ValueError(f"sandbox mission was not found: {mission_id}") from error


def _run_root(root, map_id, mission_id):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{stamp}-{map_id}-{mission_id}"
    suffix = 1
    while path.exists():
        path = Path(root) / f"{stamp}-{map_id}-{mission_id}-{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _flight_command(revision_root, mission, events_path):
    task = "fly_to_point" if mission.mission_type == "point_to_point" else "fly_round_trip"
    command = [
        sys.executable, "main.py", "task", "run", task, "--",
        "--obstacle-config", str(revision_root / "routes" / f"{mission.mission_id}.obstacles.json"),
        "--altitude", str(mission.altitude_m),
        "--max-speed", str(mission.speed_m_s),
        "--visual-mission-events", str(events_path),
    ]
    if mission.mission_type == "equipment_inspection":
        command.extend([
            "--visual-route",
            str(revision_root / "routes" / f"{mission.mission_id}.visual.json"),
            "--return-home",
        ])
    return command


def _completed(events_path):
    try:
        rows = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        row.get("event_type") == "mission_completed"
        and row.get("landing_confirmed") is True for row in rows
    )


def _recording_context(map_value, mission, revision_id, route_record, revision_root):
    obstacle_path = revision_root / "routes" / f"{mission.mission_id}.obstacles.json"
    target = mission.target_object_id or "map_targets"
    return {
        "recording_type": "labelled_visual_collection",
        "protocol_id": "visual-pilot-png-v3",
        "dataset_role": "development",
        "source_type": "sandbox_custom_map",
        "map_id": map_value.map_id,
        "map_revision_identity_sha256": revision_id,
        "target_id": target,
        "seed": int(revision_id[:8], 16),
        "route_id": mission.mission_id,
        "route_identity_sha256": route_record["route_identity_sha256"],
        "scenario_id": f"{map_value.map_id}-{mission.mission_id}-{revision_id[:12]}",
        "split": "development",
        "collection_plan_identity_sha256": None,
        "base_scenario_config_hash": map_value.map_identity_sha256,
        "scenario_config_hash": object_sha256({
            "map_revision": revision_id,
            "route": route_record["route_identity_sha256"],
        }),
        "world_sha256": file_sha256(revision_root / "world.sdf"),
        "planner_config_sha256": file_sha256(obstacle_path),
    }


def _recording_command(root, context_path, events_path, recording_id):
    return [
        sys.executable, "-m", "src.sandbox.map_record_worker",
        "--output", str(root / "recording"),
        "--context", str(context_path),
        "--flight-events", str(events_path),
        "--recording-id", recording_id,
    ]


def run_sandbox_map(
    project_root, maps_root, runs_root, map_id, revision_id, mission_id,
    *, display_mode="headless", startup_timeout_s=180.0, flight_timeout_s=None,
    record=False,
):
    project_root, runs_root = Path(project_root), Path(runs_root)
    store = SandboxMapStore(maps_root)
    revision_root = store.revision_root(map_id, revision_id)
    map_value = SandboxMap.from_record(json.loads((revision_root / "map.json").read_text()))
    mission = _mission(map_value, mission_id)
    route_record = json.loads(
        (revision_root / "routes" / f"{mission_id}.json").read_text()
    )
    root = _run_root(runs_root, map_id, mission_id)
    recording_id = f"sandbox-{root.name}"
    events_path = (
        root / "recording" / "flight_events.jsonl"
        if record else root / "flight_events.jsonl"
    )
    telemetry_path = root / "telemetry.csv"
    live_path = root / "live_telemetry.json"
    camera_path = root / "x500_research" / "model.sdf"
    materialize_camera_model(
        project_root / "simulation/models/x500_research/model.sdf",
        camera_path, map_value.camera_noise_stddev,
    )
    pointer = runs_root / "active.json"
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    active = {
        "map_run_schema_version": 1, "state": "running", "run_root": str(root),
        "map_id": map_id, "revision_id": revision_id, "mission_id": mission_id,
        "route_identity_sha256": route_record["route_identity_sha256"],
        "display_mode": display_mode, "started_at": started_at,
    }
    _atomic_json(pointer, active)
    context_path = root / "recording_context.json"
    if record:
        write_json(
            context_path,
            _recording_context(
                map_value, mission, revision_id, route_record, revision_root
            ),
        )
    launcher = flight = recorder = None
    status, error_message = "failed", None
    started = time.monotonic()
    try:
        prepared = {"launcher_environment": {
            "MAP_ID": "custom", "WORLD_NAME": f"sandbox_{map_id}",
            "WORLD_SRC": str((revision_root / "world.sdf").resolve()),
            "PX4_GZ_MODEL_POSE": (
                f"0,0,0,0,0,{math.radians(map_value.start_yaw_deg):g}"
            ),
            "SIM_MODEL": "x500_research",
            "RESEARCH_MODEL_SRC": str(camera_path.resolve()), "HEADLESS": "1",
        }}
        environment = os.environ.copy()
        environment.update(launcher_environment(prepared, display_mode))
        launcher = start_process(
            "PX4/Gazebo launcher", ["bash", "scripts/flight/start_px4_substation.sh"],
            root / "simulator.log", env=environment, discard_stdout=True,
        )
        ensure_process_running(launcher)
        probe_until_ready(
            root / "probe.log", startup_timeout_s=startup_timeout_s,
            probe_timeout_s=5.0, required_process=launcher,
        )
        if record:
            recorder = start_process(
                "sandbox map recorder",
                _recording_command(root, context_path, events_path, recording_id),
                root / "recorder.log",
            )
            time.sleep(1.0)
            ensure_process_running(recorder)
        flight_env = os.environ.copy()
        flight_env.update({
            "UAV_TELEMETRY_LOG_PATH": str(telemetry_path.resolve()),
            "UAV_LIVE_TELEMETRY_PATH": str(live_path.resolve()),
        })
        flight = start_process(
            "sandbox map flight", _flight_command(revision_root, mission, events_path),
            root / "flight.log", env=flight_env,
        )
        timeout_s = float(flight_timeout_s or min(
            900.0, max(120.0, 2.0 * route_record["estimated_duration_s"])
        ))
        wait_process(flight, timeout_s)
        if not _completed(events_path):
            raise CollectionProcessError("sandbox map flight ended without confirmed landing")
        if recorder is not None:
            wait_process(recorder, 30.0)
        status = "complete"
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_process(flight)
        stop_process(recorder)
        stop_process(launcher)
        summary = {
            **active, "state": status, "status": status, "error": error_message,
            "duration_s": round(time.monotonic() - started, 3),
            "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mission_completed": _completed(events_path), "code_commit": git_commit(),
            "recording_id": recording_id if record else None,
            "recording_state": "complete" if record and status == "complete" else None,
            "artifacts": {
                "telemetry": "telemetry.csv", "live_telemetry": "live_telemetry.json",
                "events": str(events_path.relative_to(root)), "flight_log": "flight.log",
                "simulator_log": "simulator.log",
                "recording": "recording" if record else None,
            },
        }
        write_json(root / "summary.json", summary)
        _atomic_json(runs_root / "latest.json", summary)
        if pointer.exists():
            pointer.unlink()
    return root
