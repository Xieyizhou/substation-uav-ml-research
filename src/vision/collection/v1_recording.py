"""Scenario materialization for visual collection protocol v1."""

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.domain_randomization import load_ranges, sample_manifest
from src.maps.map_catalog import map_by_id, project_path, spawn_pose_text
from src.study.runner import _materialize_reachable_scenario
from src.vision.collection.plan import visual_randomization_identity


def _update_report(plan, row, protocol, world_path, report_path, planner_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "collection_plan_identity_sha256": plan["collection_plan_identity_sha256"],
            "collection_protocol_id": protocol.protocol_id,
            "scenario_id": row["scenario_id"],
            "target_id": row["target_id"],
            "route_id": row["route_id"],
            "split": row["split"],
            "base_scenario_config_hash": row["base_scenario_config_hash"],
            "applied_visual_randomization_fields": protocol["randomization"]["applied_visual_fields"],
            "randomization_fields_not_applied_to_visuals": protocol["randomization"]["explicitly_not_applied"],
        }
    )
    report["visual_scenario_config_hash"] = object_sha256(
        {
            "applied_visual_randomization_fields": report["applied_visual_randomization_fields"],
            "equipment_changes": report["equipment_changes"],
            "light_intensity": report["light_intensity"],
            "unknown_obstacles": report["unknown_obstacles"],
            "feasibility_adjustments": report.get("feasibility_adjustments", []),
        }
    )
    report["world_sha256"] = file_sha256(world_path)
    report["planner_config_sha256"] = file_sha256(planner_path)
    write_json(report_path, report)


def _runtime_record(output_root, row, entry, paths):
    world_path, report_path, planner_path = paths
    recording_directory = Path(output_root) / "recordings" / row["recording_id"]
    flight_events_path = recording_directory / "flight_events.jsonl"
    return {
        "scenario": row,
        "scenario_report": str(report_path),
        "world_path": str(world_path),
        "planner_path": str(planner_path),
        "recording_directory": str(recording_directory),
        "launcher_environment": {
            "MAP_ID": "custom",
            "WORLD_NAME": entry["world_name"],
            "WORLD_SRC": str(world_path.resolve()),
            "PX4_GZ_MODEL_POSE": spawn_pose_text(entry),
            "SIM_MODEL": "x500_research",
            "HEADLESS": "1",
        },
        "launcher_command": ["bash", "scripts/flight/start_px4_substation.sh"],
        "flight_command": [
            "python", "main.py", "task", "run", "fly_round_trip", "--",
            "--obstacle-config", str(planner_path),
            "--visual-mission-events", str(flight_events_path),
        ],
        "flight_events_path": str(flight_events_path),
    }


def prepare_v1_collection_scenario(plan, row, output_root, protocol):
    entry = map_by_id(row["map_id"])
    randomization = load_ranges(
        project_path(protocol["randomization"]["configuration"])
    )
    manifest = sample_manifest(
        randomization,
        map_id=row["map_id"],
        seed=row["seed"],
    )
    if visual_randomization_identity(manifest, protocol) != row["base_scenario_config_hash"]:
        raise ValueError("scenario randomization does not match collection plan")
    root = Path(output_root) / "scenarios" / row["scenario_id"]
    paths = (root / "world.sdf", root / "scenario.json", root / "planner.json")
    _materialize_reachable_scenario(
        manifest,
        entry,
        row["target_id"],
        *paths,
    )
    _update_report(plan, row, protocol, *paths)
    return _runtime_record(output_root, row, entry, paths)
