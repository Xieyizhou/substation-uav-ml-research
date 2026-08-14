"""Materialize terminal flight outcomes without hiding safe mission failures."""

from __future__ import annotations

import json
from pathlib import Path

from src.logging.analysis_summaries import perception_summary, replan_summary
from src.logging.collision_checks import obstacle_collision_report
from src.logging.log_io import prepare_dataframe
from src.ml.artifacts import file_sha256, git_commit
from src.planner.obstacle_config import build_obstacle_map
from src.study.quality_metrics import lidar_quality_metrics
from src.vision.collection.process import CollectionProcessError


MISSION_FAILURE_MARKERS = (
    "timed out before reaching",
    "waypoint timeout",
    "dangerobstacledetected",
    "route blocked",
    "no safe path",
)


def landed_mission_status(log_path):
    value = json.loads(
        Path(log_path).with_suffix(".status.json").read_text(encoding="utf-8")
    )
    if value.get("status") not in {"completed", "failed"}:
        raise CollectionProcessError("flight ended without a terminal mission status")
    if value.get("landing_confirmed") is not True:
        raise CollectionProcessError("flight ended without confirmed landing")
    return value


def mission_outcome_class(mission_status):
    if mission_status.get("status") == "completed":
        return "mission_success"
    message = str(mission_status.get("message", "")).lower()
    if any(marker in message for marker in MISSION_FAILURE_MARKERS):
        return "mission_failure"
    raise CollectionProcessError(
        "failed flight is not a classified mission outcome: "
        + (message or "missing failure reason")
    )


def mission_metrics(log_path, planner_path, mission_status):
    frame = prepare_dataframe(log_path)
    perception = perception_summary(frame)
    replanning = replan_summary(frame) or {}
    planner = json.loads(Path(planner_path).read_text(encoding="utf-8"))
    collision = obstacle_collision_report(
        frame, build_obstacle_map(planner), float(planner.get("resolution_m", 1.0))
    )
    health = perception.get("sensor_healthy_ratio")
    return {
        "mission_success": int(mission_status["status"] == "completed"),
        "landing_success": 1,
        "collision_count": int(collision["raw_physical_collision_detected"]),
        "buffer_entry_count": int(collision["inflated_safety_buffer_entry_detected"]),
        "safety_failure_count": int(health is None or health < 0.99),
        "flight_time_s": float(frame["elapsed_s"].max()) if not frame.empty else 0.0,
        "sensor_healthy_ratio": health,
        "inference_p95_ms": perception.get("inference_latency_p95_ms"),
        "replan_attempt_count": replanning.get("total_replan_attempts"),
        "successful_replan_count": replanning.get("successful_replan_attempts"),
        "active_replan_count": replanning.get("active_route_replacement_count"),
        **lidar_quality_metrics(frame),
    }


def result_payload(row, log_path, metrics, mission_status, formal_receipt):
    result = {
        "schema_version": 1,
        "run_id": row["run_id"],
        "scenario_id": row["scenario_id"],
        "condition": row["condition"],
        "code_commit": git_commit(),
        "metrics": metrics,
        "mission": {
            "status": mission_status["status"],
            "outcome_class": mission_outcome_class(mission_status),
            "message": mission_status.get("message", ""),
            "landing_confirmed": mission_status["landing_confirmed"],
        },
        "artifacts": [{
            "kind": "flight_log",
            "path": str(log_path),
            "sha256": file_sha256(log_path),
        }],
    }
    if formal_receipt is not None:
        result["formal_study_identity_sha256"] = formal_receipt[
            "formal_study_identity_sha256"
        ]
    return result
