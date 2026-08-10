"""Run one visual flight without recording dataset frames."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from src.ml.artifacts import git_commit, write_json
from src.vision.collection.plan import collection_status, load_collection_plan
from src.vision.collection.process import (
    CollectionProcessError,
    ensure_process_running,
    probe_until_ready,
    start_process,
    stop_process,
    wait_process,
)
from src.vision.collection.recording import (
    prepare_collection_scenario,
    scenario_by_id,
)


BLIND_ROLES = frozenset({"blind", "held_out_test", "test"})


def choose_scenario(plan, output_root, scenario_id=None):
    if scenario_id is not None:
        row = scenario_by_id(plan, scenario_id)
    else:
        status = collection_status(plan, Path(output_root) / "recordings")
        row = status.get("next_scenario")
        if row is None:
            row = plan["scenarios"][0]
    role = str(row.get("dataset_role", row.get("split", ""))).lower()
    if role in BLIND_ROLES:
        raise ValueError("flight smoke cannot expose a blind scenario")
    return row


def _new_run_root(root, scenario_id):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = Path(root) / f"{stamp}-{scenario_id}"
    suffix = 1
    while candidate.exists():
        candidate = Path(root) / f"{stamp}-{scenario_id}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _flight_command(prepared, events_path):
    command = [sys.executable, *prepared["flight_command"][1:]]
    index = command.index("--visual-mission-events") + 1
    command[index] = str(events_path)
    return command


def _mission_completed(events_path):
    try:
        rows = [
            json.loads(line)
            for line in Path(events_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        row.get("event_type") == "mission_completed"
        and row.get("landing_confirmed") is True
        for row in rows
    )


def run_flight_smoke(
    plan_path,
    output_root,
    run_root,
    *,
    scenario_id=None,
    startup_timeout_s=180.0,
    probe_timeout_s=5.0,
    flight_timeout_s=None,
):
    plan = load_collection_plan(plan_path)
    row = choose_scenario(plan, output_root, scenario_id)
    prepared = prepare_collection_scenario(plan, row["scenario_id"], output_root)
    root = _new_run_root(run_root, row["scenario_id"])
    events_path = root / "flight_events.jsonl"
    launcher = flight = None
    started = time.monotonic()
    status = "failed"
    error_message = ""
    try:
        environment = os.environ.copy()
        environment.update(prepared["launcher_environment"])
        launcher = start_process(
            "PX4/Gazebo launcher",
            prepared["launcher_command"],
            root / "simulator.log",
            env=environment,
            discard_stdout=True,
        )
        ensure_process_running(launcher)
        probe_until_ready(
            root / "probe.log",
            startup_timeout_s=startup_timeout_s,
            probe_timeout_s=probe_timeout_s,
            required_process=launcher,
        )
        flight = start_process(
            "flight task",
            _flight_command(prepared, events_path),
            root / "flight.log",
        )
        timeout_s = float(
            flight_timeout_s
            if flight_timeout_s is not None
            else prepared.get("flight_timeout_s", 360.0)
        )
        wait_process(flight, timeout_s)
        if not _mission_completed(events_path):
            raise CollectionProcessError(
                "flight smoke ended without confirmed mission completion"
            )
        status = "complete"
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_process(flight)
        stop_process(launcher)
        write_json(root / "summary.json", {
            "flight_smoke_schema_version": 1,
            "run_type": "sandbox_flight_smoke",
            "status": status,
            "error": error_message or None,
            "scenario_id": row["scenario_id"],
            "dataset_role": row.get("dataset_role"),
            "duration_s": round(time.monotonic() - started, 3),
            "mission_completed": _mission_completed(events_path),
            "code_commit": git_commit(),
            "logs": {
                "simulator": "simulator.log",
                "probe": "probe.log",
                "flight": "flight.log",
                "events": "flight_events.jsonl",
            },
        })
    return root
