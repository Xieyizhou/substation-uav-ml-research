"""Resumable local process orchestration for visual scenario collection."""

from __future__ import annotations

from pathlib import Path
import shutil
import time

from src.vision.collection.plan import collection_status
from src.vision.collection.process import CollectionProcessError
from src.vision.collection.recording import scenario_by_id
from src.vision.collection.scenario_runner import (
    ensure_validated_receipt,
    run_collection_scenario,
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


def _dry_run_batch(plan, recordings_root, max_scenarios):
    remaining = plan
    processed = []
    while max_scenarios is None or len(processed) < max_scenarios:
        next_row = collection_status(remaining, recordings_root)["next_scenario"]
        if next_row is None:
            break
        if next_row["state"] not in {"missing", "unvalidated"}:
            raise CollectionBatchError(
                f"{next_row['scenario_id']} is {next_row['state']}; inspect "
                f"{next_row['recording_directory']} before resuming"
            )
        processed.append(
            {
                "scenario_id": next_row["scenario_id"],
                "action": (
                    "validate" if next_row["state"] == "unvalidated" else "record"
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
    return processed


def _run_attempts(plan, plan_path, output_root, row, max_attempts, delay_s, options):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = run_collection_scenario(
                plan,
                plan_path,
                row["scenario_id"],
                output_root,
                **options,
            )
            return {**result, "attempt": attempt}
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
                time.sleep(delay_s)
    raise CollectionBatchError(
        f"{row['scenario_id']} failed after {max_attempts} attempts"
    ) from last_error


def _process_next(plan, plan_path, output_root, recordings_root, attempts, delay, options):
    next_row = collection_status(plan, recordings_root)["next_scenario"]
    if next_row is None:
        return None
    row = scenario_by_id(plan, next_row["scenario_id"])
    if next_row["state"] == "unvalidated":
        ensure_validated_receipt(next_row["recording_directory"], row, plan)
        return {"scenario_id": row["scenario_id"], "state": "complete"}
    if next_row["state"] != "missing":
        raise CollectionBatchError(
            f"{row['scenario_id']} is {next_row['state']}; inspect "
            f"{next_row['recording_directory']} before resuming"
        )
    print(f"Starting visual collection scenario {row['scenario_id']}", flush=True)
    return _run_attempts(
        plan,
        plan_path,
        output_root,
        row,
        attempts,
        delay,
        options,
    )


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
        completed = _dry_run_batch(plan, recordings_root, max_scenarios)
        return {
            "processed": completed,
            "processed_count": len(completed),
            "status": collection_status(plan, recordings_root),
        }
    while max_scenarios is None or len(completed) < max_scenarios:
        result = _process_next(
            plan,
            plan_path,
            output_root,
            recordings_root,
            max_attempts,
            retry_delay_s,
            scenario_options,
        )
        if result is None:
            break
        completed.append(result)
    return {
        "processed": completed,
        "processed_count": len(completed),
        "status": collection_status(plan, recordings_root),
    }
