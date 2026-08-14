"""Fail-fast local worker for sequential LiDAR study flight tiers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.ml.artifacts import object_sha256, write_json
from src.study.flight_budget import closed_loop_timeout_s
from src.study.flight_progress_watchdog import wait_for_study_flight
from src.study.formal_spec import DEFAULT_FORMAL_SPEC, freeze_formal_study
from src.study.comparison import closed_loop_gate
from src.study.challenge_receipt import inspect_challenge_receipt
from src.study.mission_result import (
    landed_mission_status,
    mission_metrics,
    mission_outcome_class,
    result_payload,
)
from src.study.registry import ResearchRegistry
from src.study.runner import ingest_results
from src.vision.collection.process import (
    CollectionProcessError,
    ensure_process_running,
    start_process,
    stop_process,
)
ROOT = Path(__file__).resolve().parents[2]
FLIGHT_TIERS = frozenset({"challenge", "closed-loop", "formal"})
STARTUP_RUN_ATTEMPTS = 2


def _attempt_root(run_root):
    attempts = Path(run_root) / "attempts"
    number = 1
    while (attempts / f"attempt_{number:02d}").exists():
        number += 1
    path = attempts / f"attempt_{number:02d}"
    path.mkdir(parents=True)
    return path


def _run_setup(commands, log_path):
    with Path(log_path).open("a", encoding="utf-8") as output:
        for command in commands:
            resolved = [sys.executable, *command[1:]] if command[0] == "python" else command
            output.write(f"COMMAND: {' '.join(str(item) for item in resolved)}\n")
            output.flush()
            subprocess.run(
                resolved, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True
            )


def _probe_lidar(log_path, launcher, startup_timeout_s, probe_timeout_s):
    deadline = time.monotonic() + startup_timeout_s
    command = [
        sys.executable, "main.py", "sensor", "list", "--json",
    ]
    while time.monotonic() < deadline:
        ensure_process_running(launcher, settle_s=0.0)
        with Path(log_path).open("a", encoding="utf-8") as output:
            result = subprocess.run(
                command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=probe_timeout_s + 10.0,
                text=True,
            )
            output.write(result.stdout or "")
        if result.returncode == 0:
            for line in reversed((result.stdout or "").splitlines()):
                try:
                    topic = json.loads(line).get("lidar_2d")
                except json.JSONDecodeError:
                    continue
                if topic:
                    return topic
        time.sleep(2.0)
    raise CollectionProcessError("Gazebo LiDAR was not ready before startup timeout")


def _new_flight_log(previous):
    candidates = set((ROOT / "data/logs").glob("astar_*.csv")) - previous
    if len(candidates) != 1:
        raise CollectionProcessError(
            f"expected one new flight log, found {len(candidates)}"
        )
    return candidates.pop()


def _is_retryable_startup_failure(error, attempt_root):
    """Retry only failures that occurred before a telemetry log was created."""
    if not isinstance(error, CollectionProcessError):
        return False
    try:
        flight_output = (Path(attempt_root) / "flight.log").read_text(
            encoding="utf-8"
        )
    except OSError:
        return False
    transport_failed = "MAVSDK connection failed" in flight_output
    first_scan_missing = "did not become ready within" in flight_output and (
        "waiting for first scan" in flight_output
    )
    return "expected one new flight log, found 0" in str(error) and (transport_failed or first_scan_missing)


def _run_one(
    row, run_root, *, startup_timeout_s, probe_timeout_s, flight_timeout_s,
    allow_progress_extension=False,
):
    run_root.mkdir(parents=True, exist_ok=True)
    launcher = flight = None
    before = set((ROOT / "data/logs").glob("astar_*.csv"))
    try:
        _run_setup(row["setup_commands"], run_root / "setup.log")
        environment = os.environ.copy()
        environment.update(row["launcher_environment"])
        launcher = start_process(
            "PX4/Gazebo launcher", row["launcher_command"],
            run_root / "simulator.log", env=environment,
            discard_stdout=True,
        )
        ensure_process_running(launcher)
        lidar_topic = _probe_lidar(
            run_root / "probe.log", launcher, startup_timeout_s, probe_timeout_s
        )
        command = [
            sys.executable, *row["flight_command"][1:],
            "--sensor-topic", lidar_topic,
        ]
        flight = start_process("flight task", command, run_root / "flight.log")
        process_error = None
        try:
            wait_for_study_flight(
                flight, flight_timeout_s, before_logs=before,
                allow_progress_extension=allow_progress_extension,
            )
        except CollectionProcessError as error:
            process_error = error
        log_path = _new_flight_log(before)
        mission_status = landed_mission_status(log_path)
        outcome_class = mission_outcome_class(mission_status)
        if process_error is not None and outcome_class != "mission_failure":
            raise process_error
        metrics = mission_metrics(
            log_path, row["oracle_planner_config"], mission_status
        )
        return log_path, metrics, mission_status
    finally:
        stop_process(flight)
        stop_process(launcher)


def _verify_replay_receipt(
    registry, study_id, path, qualification_study_id=None
):
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = receipt.pop("replay_gate_identity_sha256", None)
    if supplied != object_sha256(receipt):
        raise ValueError("replay gate identity mismatch")
    receipt["replay_gate_identity_sha256"] = supplied
    study = registry.get_study(study_id)
    qualification_study_id = qualification_study_id or study_id
    qualification = registry.get_study(qualification_study_id)
    if qualification["candidate_model"] != study["candidate_model"]:
        raise ValueError("qualification study candidate does not match formal study")
    model = registry.get_model(study["candidate_model"])
    manifest = json.loads(model["manifest_json"])
    if receipt.get("passed") is not True:
        raise ValueError("formal execution requires a passed replay gate")
    if receipt.get("model_id") != study["candidate_model"]:
        raise ValueError("replay gate model does not match the study candidate")
    if receipt.get("model_sha256") != model["onnx_hash"]:
        raise ValueError("replay gate ONNX hash does not match the study candidate")
    if receipt.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("replay gate dataset does not match the study candidate")
    closed = registry.run_metrics(qualification_study_id, "closed-loop")
    decision = closed_loop_gate(closed)
    if not decision["passed"]:
        details = "; ".join(decision.get("reasons", []))
        raise ValueError(
            "formal execution requires demonstrated capability coverage: " + details
        )
    return receipt


def _prepare_tier(
    registry, study_id, results_dir, tier, replay_gate_path, comparison_spec_path,
    qualification_study_id, flight_timeout_s, challenge_receipt_path,
):
    if tier != "formal":
        return None
    if replay_gate_path is None:
        raise ValueError("formal execution requires --replay-gate")
    model_id = registry.get_study(study_id)["candidate_model"]
    candidates = ([Path(challenge_receipt_path)] if challenge_receipt_path else sorted(
        Path(results_dir).glob("*/challenge/challenge_receipt.json"),
        key=lambda path: path.stat().st_mtime_ns, reverse=True,
    ))
    challenge = None
    for path in candidates:
        try:
            value = inspect_challenge_receipt(
                path, project_root=ROOT, registry_path=registry.path,
                require_current=True,
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
        if value.get("model_id") == model_id and value.get("passed") is True:
            challenge = value
            break
    if challenge is None:
        raise ValueError(
            "formal execution requires a current passed capability challenge receipt"
        )
    qualification_study_id = qualification_study_id or study_id
    replay = _verify_replay_receipt(
        registry, study_id, replay_gate_path, qualification_study_id
    )
    return freeze_formal_study(
        registry, study_id, results_dir, replay, comparison_spec_path,
        qualification_study_id=qualification_study_id,
        flight_timeout_override_s=flight_timeout_s,
    )


def _execute_row(
    registry, row, run_base, tier, results_dir, formal_receipt, *,
    startup_timeout_s, probe_timeout_s, flight_timeout_s,
):
    registry.set_run_status(row["run_id"], "running")
    for attempt_number in range(1, STARTUP_RUN_ATTEMPTS + 1):
        run_root = _attempt_root(run_base)
        try:
            timeout_s = closed_loop_timeout_s(
                row["oracle_planner_config"], flight_timeout_s
            )
            log_path, metrics, mission_status = _run_one(
                row, run_root, startup_timeout_s=startup_timeout_s,
                probe_timeout_s=probe_timeout_s, flight_timeout_s=timeout_s,
                allow_progress_extension=flight_timeout_s is None,
            )
            write_json(
                row["result_path"],
                result_payload(
                    row, log_path, metrics, mission_status, formal_receipt
                ),
            )
            ingest_results(registry, row["study_id"], tier, results_dir)
            return
        except Exception as error:
            write_json(run_root / "failure.json", {
                "run_id": row["run_id"], "scenario_id": row["scenario_id"],
                "condition": row["condition"],
                "error": f"{type(error).__name__}: {error}",
            })
            if (
                attempt_number < STARTUP_RUN_ATTEMPTS
                and _is_retryable_startup_failure(error, run_root)
            ):
                continue
            registry.set_run_status(
                row["run_id"], "failed",
                failure_reason=f"{type(error).__name__}: {error}",
            )
            raise


def execute_flight_tier(
    registry_path, study_id, results_dir, *, tier, max_runs=None,
    startup_timeout_s=180.0, probe_timeout_s=5.0, flight_timeout_s=None,
    replay_gate_path=None,
    comparison_spec_path=DEFAULT_FORMAL_SPEC,
    qualification_study_id=None,
    challenge_receipt_path=None,
    scenario_id=None,
):
    """Execute one flight tier sequentially and stop on the first failure."""
    if tier not in FLIGHT_TIERS:
        raise ValueError(f"unsupported flight tier: {tier}")
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max runs must be positive")
    registry = ResearchRegistry(registry_path)
    formal_receipt = _prepare_tier(
        registry, study_id, results_dir, tier, replay_gate_path,
        comparison_spec_path, qualification_study_id, flight_timeout_s,
        challenge_receipt_path,
    )
    scheduled = ingest_results(registry, study_id, tier, results_dir)
    queue = json.loads(Path(scheduled["run_queue"]).read_text(encoding="utf-8"))
    pending = [row for row in queue["runs"] if row["status"] != "completed"]
    if scenario_id is not None:
        pending = [row for row in pending if row["scenario_id"] == scenario_id]
        if not pending:
            raise ValueError(f"no pending runs match scenario {scenario_id!r}")
    selected = pending[:max_runs] if max_runs is not None else pending
    completed = []
    for row in selected:
        row["study_id"] = study_id
        run_base = Path(results_dir) / study_id / tier / "runs" / row["run_id"]
        _execute_row(
            registry, row, run_base, tier, results_dir, formal_receipt,
            startup_timeout_s=startup_timeout_s,
            probe_timeout_s=probe_timeout_s,
            flight_timeout_s=flight_timeout_s,
        )
        completed.append(row["run_id"])
    status = ingest_results(registry, study_id, tier, results_dir)
    return {**status, "executed": len(completed), "run_ids": completed}


def execute_closed_loop(registry_path, study_id, results_dir, **options):
    return execute_flight_tier(
        registry_path, study_id, results_dir, tier="closed-loop", **options
    )


def execute_formal(registry_path, study_id, results_dir, **options):
    return execute_flight_tier(
        registry_path, study_id, results_dir, tier="formal", **options
    )
