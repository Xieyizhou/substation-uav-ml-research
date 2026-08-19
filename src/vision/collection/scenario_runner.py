"""Runtime orchestration for one visual collection scenario."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from src.vision.collection.process import (
    CollectionProcessError,
    ensure_process_running,
    probe_until_ready,
    start_process,
    stop_process,
    wait_for_jsonl_event,
    wait_for_flight,
    wait_process,
)
from src.vision.collection.display import launcher_environment
from src.vision.collection.receipt import (
    collection_validation_receipt_is_current,
    write_collection_validation_receipt,
)
from src.vision.collection.recording import (
    prepare_collection_scenario,
    scenario_by_id,
    validate_collection_recording,
)


def ensure_validated_receipt(recording_directory, row, plan):
    if collection_validation_receipt_is_current(recording_directory, row, plan):
        return
    validation = validate_collection_recording(recording_directory, plan)
    write_collection_validation_receipt(recording_directory, plan, validation)


def _wait_for_first_frame(recording_directory, recorder, timeout_s):
    status_path = Path(recording_directory) / "live_status.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if recorder.process.poll() is not None:
            recorder.close_log()
            raise CollectionProcessError(
                f"recorder exited before the first frame; inspect {recorder.log_path}"
            )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            status = {}
        if int(status.get("accepted_frame_count", 0)) > 0:
            return
        time.sleep(0.2)
    raise CollectionProcessError(
        f"recorder received no frame within {timeout_s:.0f}s; "
        f"inspect {recorder.log_path}"
    )


def _start_simulator(
    prepared, log_root, startup_timeout_s, probe_timeout_s, display_mode,
):
    environment = os.environ.copy()
    environment.update(launcher_environment(prepared, display_mode))
    launcher = start_process(
        "PX4/Gazebo launcher",
        prepared["launcher_command"],
        log_root / "simulator.log",
        env=environment,
        discard_stdout=True,
    )
    ensure_process_running(launcher)
    probe_until_ready(
        log_root / "probe.log",
        startup_timeout_s=startup_timeout_s,
        probe_timeout_s=probe_timeout_s,
        required_process=launcher,
    )
    return launcher


def _start_recorder(prepared, plan_path, scenario_id, output_root, log_root):
    command = [
        sys.executable, "main.py", "visual", "collection-record",
        "--plan", str(plan_path),
        "--scenario-id", scenario_id,
        "--output-root", str(output_root),
        "--scenario-report", prepared["scenario_report"],
    ]
    return start_process("visual recorder", command, log_root / "recorder.log")


def run_collection_scenario(
    plan, plan_path, scenario_id, output_root, *,
    simulator_startup_timeout_s=180.0, probe_timeout_s=5.0,
    takeoff_ready_timeout_s=45.0, first_frame_timeout_s=30.0,
    flight_timeout_s=None,
    recorder_timeout_s=900.0,
    display_mode="headless",
):
    prepared = prepare_collection_scenario(plan, scenario_id, output_root)
    row = scenario_by_id(plan, scenario_id)
    recording_directory = Path(prepared["recording_directory"])
    log_root = Path(output_root) / "batch_logs" / scenario_id
    launcher = recorder = flight = None
    try:
        launcher = _start_simulator(
            prepared, log_root, simulator_startup_timeout_s, probe_timeout_s,
            display_mode,
        )
        flight = start_process(
            "flight task",
            [sys.executable, *prepared["flight_command"][1:]],
            log_root / "flight.log",
        )
        wait_for_jsonl_event(
            prepared["flight_events_path"],
            "takeoff_completed",
            flight,
            takeoff_ready_timeout_s,
        )
        recorder = _start_recorder(
            prepared, plan_path, scenario_id, output_root, log_root
        )
        _wait_for_first_frame(recording_directory, recorder, first_frame_timeout_s)
        timeout_s = (
            float(flight_timeout_s) if flight_timeout_s is not None
            else float(prepared.get("flight_timeout_s", 360.0))
        )
        wait_for_flight(flight, recorder, timeout_s)
        wait_process(recorder, recorder_timeout_s)
        ensure_validated_receipt(recording_directory, row, plan)
        return {
            "scenario_id": scenario_id,
            "recording_id": row["recording_id"],
            "state": "complete",
            "display_mode": display_mode,
            "log_directory": str(log_root),
        }
    finally:
        stop_process(flight)
        stop_process(recorder)
        stop_process(launcher)
