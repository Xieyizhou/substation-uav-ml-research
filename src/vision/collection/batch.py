"""Resumable local process orchestration for visual scenario collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

from src.vision.collection.plan import collection_status
from src.vision.collection.receipt import (
    collection_validation_receipt_is_current,
    write_collection_validation_receipt,
)
from src.vision.collection.recording import (
    prepare_collection_scenario,
    scenario_by_id,
    validate_collection_recording,
)
from src.vision.collection.process import (
    CollectionProcessError,
    ensure_process_running,
    probe_until_ready,
    start_process,
    stop_process,
    wait_for_flight,
    wait_process,
)


class CollectionBatchError(CollectionProcessError):
    pass


def archive_failed_attempt(output_root, scenario_id, recording_id):
    output_root = Path(output_root)
    retry_root = output_root / "archive" / "retries" / scenario_id
    attempt_number = 1
    while (retry_root / f"attempt_{attempt_number:02d}").exists():
        attempt_number += 1
    destination = retry_root / f"attempt_{attempt_number:02d}"
    recording_directory = (
        output_root
        / "recordings"
        / recording_id
    )
    log_directory = output_root / "batch_logs" / scenario_id
    destination.mkdir(parents=True)
    if recording_directory.exists():
        shutil.move(str(recording_directory), destination / "recording")
    if log_directory.exists():
        shutil.move(str(log_directory), destination / "logs")
    return destination


def _wait_for_first_frame(recording_directory, recorder, timeout_s):
    status_path = Path(recording_directory) / "live_status.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if recorder.process.poll() is not None:
            recorder.close_log()
            raise CollectionBatchError(
                f"recorder exited before the first frame; inspect {recorder.log_path}"
            )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            status = {}
        if int(status.get("accepted_frame_count", 0)) > 0:
            return
        time.sleep(0.2)
    raise CollectionBatchError(
        f"recorder received no frame within {timeout_s:.0f}s; "
        f"inspect {recorder.log_path}"
    )


def _validated_receipt(recording_directory, row, plan):
    if collection_validation_receipt_is_current(recording_directory, row, plan):
        return
    validation = validate_collection_recording(recording_directory, plan)
    write_collection_validation_receipt(
        recording_directory,
        plan,
        validation,
    )


def run_collection_scenario(
    plan,
    plan_path,
    scenario_id,
    output_root,
    *,
    simulator_startup_timeout_s=180.0,
    probe_timeout_s=5.0,
    first_frame_timeout_s=30.0,
    flight_timeout_s=600.0,
    recorder_timeout_s=900.0,
):
    prepared = prepare_collection_scenario(plan, scenario_id, output_root)
    row = scenario_by_id(plan, scenario_id)
    recording_directory = Path(prepared["recording_directory"])
    log_root = Path(output_root) / "batch_logs" / scenario_id
    launcher = recorder = flight = None
    try:
        launcher_env = os.environ.copy()
        launcher_env.update(prepared["launcher_environment"])
        launcher = start_process(
            "PX4/Gazebo launcher",
            prepared["launcher_command"],
            log_root / "simulator.log",
            env=launcher_env,
            discard_stdout=True,
        )
        ensure_process_running(launcher)
        probe_until_ready(
            log_root / "probe.log",
            startup_timeout_s=simulator_startup_timeout_s,
            probe_timeout_s=probe_timeout_s,
            required_process=launcher,
        )
        recorder_command = [
            sys.executable,
            "main.py",
            "visual",
            "collection-record",
            "--plan",
            str(plan_path),
            "--scenario-id",
            scenario_id,
            "--scenario-report",
            prepared["scenario_report"],
        ]
        recorder = start_process(
            "visual recorder",
            recorder_command,
            log_root / "recorder.log",
        )
        _wait_for_first_frame(
            recording_directory,
            recorder,
            first_frame_timeout_s,
        )
        flight_command = [sys.executable, *prepared["flight_command"][1:]]
        flight = start_process(
            "flight task",
            flight_command,
            log_root / "flight.log",
        )
        wait_for_flight(flight, recorder, flight_timeout_s)
        flight = None
        wait_process(recorder, recorder_timeout_s)
        recorder = None
        _validated_receipt(recording_directory, row, plan)
        return {
            "scenario_id": scenario_id,
            "recording_id": row["recording_id"],
            "state": "complete",
            "log_directory": str(log_root),
        }
    finally:
        stop_process(flight)
        stop_process(recorder)
        stop_process(launcher)


def run_collection_batch(
    plan,
    plan_path,
    output_root,
    *,
    max_scenarios=None,
    dry_run=False,
    max_attempts=3,
    retry_delay_s=10.0,
    **scenario_options,
):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_delay_s < 0:
        raise ValueError("retry_delay_s must be non-negative")
    recordings_root = Path(output_root) / "recordings"
    completed = []
    if dry_run:
        remaining = plan
        while max_scenarios is None or len(completed) < max_scenarios:
            next_row = collection_status(remaining, recordings_root)[
                "next_scenario"
            ]
            if next_row is None:
                break
            if next_row["state"] not in {"missing", "unvalidated"}:
                raise CollectionBatchError(
                    f"{next_row['scenario_id']} is {next_row['state']}; inspect "
                    f"{next_row['recording_directory']} before resuming"
                )
            completed.append(
                {
                    "scenario_id": next_row["scenario_id"],
                    "action": (
                        "validate"
                        if next_row["state"] == "unvalidated"
                        else "record"
                    ),
                }
            )
            remaining = {
                **remaining,
                "scenarios": [
                    row
                    for row in remaining["scenarios"]
                    if row["scenario_id"] != next_row["scenario_id"]
                ],
            }
        return {
            "processed": completed,
            "processed_count": len(completed),
            "status": collection_status(plan, recordings_root),
        }

    while max_scenarios is None or len(completed) < max_scenarios:
        status = collection_status(plan, recordings_root)
        next_row = status["next_scenario"]
        if next_row is None:
            break
        state = next_row["state"]
        row = scenario_by_id(plan, next_row["scenario_id"])
        if state == "unvalidated":
            _validated_receipt(next_row["recording_directory"], row, plan)
            completed.append(
                {"scenario_id": row["scenario_id"], "state": "complete"}
            )
            continue
        if state != "missing":
            raise CollectionBatchError(
                f"{row['scenario_id']} is {state}; inspect "
                f"{next_row['recording_directory']} before resuming"
            )
        print(f"Starting visual collection scenario {row['scenario_id']}", flush=True)
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = run_collection_scenario(
                    plan,
                    plan_path,
                    row["scenario_id"],
                    output_root,
                    **scenario_options,
                )
                result["attempt"] = attempt
                completed.append(result)
                last_error = None
                break
            except CollectionProcessError as error:
                last_error = error
                archived = archive_failed_attempt(
                    output_root,
                    row["scenario_id"],
                    row["recording_id"],
                )
                print(
                    f"{row['scenario_id']} attempt {attempt} failed: {error}. "
                    f"Archived at {archived}",
                    flush=True,
                )
                if attempt < max_attempts:
                    time.sleep(retry_delay_s)
        if last_error is not None:
            raise CollectionBatchError(
                f"{row['scenario_id']} failed after {max_attempts} attempts"
            ) from last_error
    return {
        "processed": completed,
        "processed_count": len(completed),
        "status": collection_status(plan, recordings_root),
    }
