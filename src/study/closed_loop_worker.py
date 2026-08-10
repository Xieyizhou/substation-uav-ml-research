"""Fail-fast local worker for the five-scenario LiDAR closed-loop gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.logging.analysis_summaries import perception_summary, replan_summary
from src.logging.collision_checks import obstacle_collision_report
from src.logging.log_io import prepare_dataframe
from src.ml.artifacts import file_sha256, git_commit, write_json
from src.planner.obstacle_config import build_obstacle_map
from src.study.registry import ResearchRegistry
from src.study.runner import ingest_results
from src.vision.collection.process import (
    CollectionProcessError,
    ensure_process_running,
    start_process,
    stop_process,
    wait_process,
)


ROOT = Path(__file__).resolve().parents[2]


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
        sys.executable, "main.py", "sensor", "check", "--source",
        "gazebo_lidar_2d", "--timeout", str(probe_timeout_s),
    ]
    while time.monotonic() < deadline:
        ensure_process_running(launcher, settle_s=0.0)
        with Path(log_path).open("a", encoding="utf-8") as output:
            result = subprocess.run(
                command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                timeout=probe_timeout_s + 10.0,
            )
        if result.returncode == 0:
            return
        time.sleep(2.0)
    raise CollectionProcessError("Gazebo LiDAR was not ready before startup timeout")


def _new_flight_log(previous):
    candidates = set((ROOT / "data/logs").glob("astar_*.csv")) - previous
    if len(candidates) != 1:
        raise CollectionProcessError(
            f"expected one new flight log, found {len(candidates)}"
        )
    return candidates.pop()


def _completed_status(log_path):
    path = log_path.with_suffix(".status.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed" or value.get("landing_confirmed") is not True:
        raise CollectionProcessError("flight ended without confirmed landing")
    return value


def _metrics(log_path, planner_path):
    frame = prepare_dataframe(log_path)
    perception = perception_summary(frame)
    replanning = replan_summary(frame) or {}
    planner = json.loads(Path(planner_path).read_text(encoding="utf-8"))
    collision = obstacle_collision_report(
        frame, build_obstacle_map(planner), float(planner.get("resolution_m", 1.0))
    )
    duration = float(frame["elapsed_s"].max()) if not frame.empty else 0.0
    health = perception.get("sensor_healthy_ratio")
    return {
        "mission_success": 1,
        "landing_success": 1,
        "collision_count": int(collision["raw_physical_collision_detected"]),
        "buffer_entry_count": int(collision["inflated_safety_buffer_entry_detected"]),
        "safety_failure_count": int(health is None or health < 0.99),
        "flight_time_s": duration,
        "sensor_healthy_ratio": health,
        "inference_p95_ms": perception.get("inference_latency_p95_ms"),
        "replan_attempt_count": replanning.get("total_replan_attempts"),
        "successful_replan_count": replanning.get("successful_replan_attempts"),
        "active_replan_count": replanning.get("active_route_replacement_count"),
    }


def _run_one(row, run_root, *, startup_timeout_s, probe_timeout_s, flight_timeout_s):
    run_root.mkdir(parents=True, exist_ok=True)
    launcher = flight = None
    before = set((ROOT / "data/logs").glob("astar_*.csv"))
    try:
        _run_setup(row["setup_commands"], run_root / "setup.log")
        environment = os.environ.copy()
        environment.update(row["launcher_environment"])
        launcher = start_process(
            "PX4/Gazebo launcher", row["launcher_command"],
            run_root / "simulator.log", env=environment, discard_stdout=True,
        )
        ensure_process_running(launcher)
        _probe_lidar(
            run_root / "probe.log", launcher, startup_timeout_s, probe_timeout_s
        )
        command = [sys.executable, *row["flight_command"][1:]]
        flight = start_process("flight task", command, run_root / "flight.log")
        wait_process(flight, flight_timeout_s)
        log_path = _new_flight_log(before)
        _completed_status(log_path)
        metrics = _metrics(log_path, row["oracle_planner_config"])
        return log_path, metrics
    finally:
        stop_process(flight)
        stop_process(launcher)


def execute_closed_loop(
    registry_path, study_id, results_dir, *, max_runs=None,
    startup_timeout_s=180.0, probe_timeout_s=5.0, flight_timeout_s=360.0,
):
    """Execute pending runs sequentially; stop immediately on the first failure."""
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max runs must be positive")
    registry = ResearchRegistry(registry_path)
    scheduled = ingest_results(registry, study_id, "closed-loop", results_dir)
    queue = json.loads(Path(scheduled["run_queue"]).read_text(encoding="utf-8"))
    pending = [row for row in queue["runs"] if row["status"] != "completed"]
    selected = pending[:max_runs] if max_runs is not None else pending
    completed = []
    for row in selected:
        run_root = Path(results_dir) / study_id / "closed-loop/runs" / row["run_id"]
        registry.set_run_status(row["run_id"], "running")
        try:
            log_path, metrics = _run_one(
                row, run_root, startup_timeout_s=startup_timeout_s,
                probe_timeout_s=probe_timeout_s, flight_timeout_s=flight_timeout_s,
            )
            result = {
                "schema_version": 1,
                "run_id": row["run_id"],
                "scenario_id": row["scenario_id"],
                "condition": row["condition"],
                "code_commit": git_commit(),
                "metrics": metrics,
                "artifacts": [
                    {"kind": "flight_log", "path": str(log_path),
                     "sha256": file_sha256(log_path)}
                ],
            }
            write_json(row["result_path"], result)
            ingest_results(registry, study_id, "closed-loop", results_dir)
            completed.append(row["run_id"])
        except Exception as error:
            registry.set_run_status(
                row["run_id"], "failed",
                failure_reason=f"{type(error).__name__}: {error}",
            )
            write_json(run_root / "failure.json", {
                "run_id": row["run_id"], "scenario_id": row["scenario_id"],
                "condition": row["condition"], "error": f"{type(error).__name__}: {error}",
            })
            raise
    status = ingest_results(registry, study_id, "closed-loop", results_dir)
    return {**status, "executed": len(completed), "run_ids": completed}
