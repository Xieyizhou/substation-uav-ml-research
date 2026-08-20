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
from src.study.dynamic_replanning import event_chain_report
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


def _read_events(path):
    if path is None or not Path(path).is_file():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_latency_ms(events, start_type, end_type):
    start = next(
        (event for event in events if event.get("event_type") == start_type), None
    )
    if start is None or start.get("host_monotonic_ns") is None:
        return None
    end = next(
        (
            event for event in events
            if event.get("event_type") == end_type
            and event.get("host_monotonic_ns") is not None
            and event["host_monotonic_ns"] >= start["host_monotonic_ns"]
        ),
        None,
    )
    if end is None:
        return None
    return (end["host_monotonic_ns"] - start["host_monotonic_ns"]) / 1_000_000.0


def mission_metrics(log_path, planner_path, mission_status, event_path=None):
    frame = prepare_dataframe(log_path)
    perception = perception_summary(frame)
    replanning = replan_summary(frame) or {}
    planner = json.loads(Path(planner_path).read_text(encoding="utf-8"))
    collision = obstacle_collision_report(
        frame, build_obstacle_map(planner), float(planner.get("resolution_m", 1.0))
    )
    health = perception.get("sensor_healthy_ratio")
    events = _read_events(event_path)
    chain = event_chain_report(events) if events else None
    event_names = [event.get("event_type") for event in events]
    spawned_at = (
        event_names.index("dynamic_blocker_spawned")
        if "dynamic_blocker_spawned" in event_names
        else None
    )
    decisions = [
        index
        for index, name in enumerate(event_names)
        if name == "dynamic_replan_decided"
    ]
    false_replan = bool(
        decisions
        and (spawned_at is None or any(index < spawned_at for index in decisions))
    ) or len(decisions) > 1
    metrics = {
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
    if chain is not None:
        successful = bool(
            chain["complete"] and (replanning.get("active_route_replacement_count") or 0) >= 1
        )
        metrics.update({
            "successful_replan": int(successful),
            "route_switch_correct": int(successful),
            "false_replan": int(false_replan),
            "event_chain_complete": int(chain["complete"]),
            "detection_to_decision_ms": _event_latency_ms(
                events, "dynamic_blocker_detected", "dynamic_replan_decided"
            ),
            "detection_to_resume_ms": _event_latency_ms(
                events, "dynamic_blocker_detected", "dynamic_replan_resumed"
            ),
        })
    return metrics


def result_payload(
    row, log_path, metrics, mission_status, formal_receipt, event_path=None
):
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
    if event_path is not None and Path(event_path).is_file():
        result["artifacts"].append({
            "kind": "mission_events",
            "path": str(event_path),
            "sha256": file_sha256(event_path),
        })
    if formal_receipt is not None:
        result["formal_study_identity_sha256"] = formal_receipt[
            "formal_study_identity_sha256"
        ]
    return result
