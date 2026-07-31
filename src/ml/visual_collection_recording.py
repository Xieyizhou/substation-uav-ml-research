"""Scenario preparation and validation for multi-recording visual datasets."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.domain_randomization import load_ranges, sample_manifest
from src.ml.visual_collection import (
    DEFAULT_PROTOCOL,
    load_collection_protocol,
    visual_randomization_identity,
)
from src.ml.visual_pilot import (
    PilotRecordingError,
    REQUIRED_MANIFESTS,
    _read_json,
    _read_jsonl,
    _sha256,
)
from src.ml.visual_pilot_acceptance import pilot_acceptance_failures
from src.ml.visual_pilot_validation import (
    _load_and_validate_linkage,
    _validate_recording_identity,
)
from src.maps.map_catalog import (
    map_by_id,
    project_path,
    spawn_pose_text,
)
from src.study.runner import _materialize_reachable_scenario


def scenario_by_id(plan, scenario_id):
    for row in plan["scenarios"]:
        if row["scenario_id"] == scenario_id:
            return row
    raise ValueError(f"collection plan has no scenario {scenario_id!r}")


def prepare_collection_scenario(
    plan,
    scenario_id,
    output_root,
    *,
    protocol_path=DEFAULT_PROTOCOL,
):
    protocol = load_collection_protocol(protocol_path)
    row = scenario_by_id(plan, scenario_id)
    entry = map_by_id(row["map_id"])
    randomization = load_ranges(
        project_path(protocol["randomization"]["configuration"])
    )
    manifest = sample_manifest(
        randomization,
        map_id=row["map_id"],
        seed=row["seed"],
    )
    if (
        visual_randomization_identity(manifest, protocol)
        != row["base_scenario_config_hash"]
    ):
        raise ValueError("scenario randomization does not match collection plan")
    root = Path(output_root) / "scenarios" / scenario_id
    world_path = root / "world.sdf"
    report_path = root / "scenario.json"
    planner_path = root / "planner.json"
    _materialize_reachable_scenario(
        manifest,
        entry,
        row["target_id"],
        world_path,
        report_path,
        planner_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "collection_plan_identity_sha256": plan[
                "collection_plan_identity_sha256"
            ],
            "collection_protocol_id": protocol["protocol_id"],
            "scenario_id": row["scenario_id"],
            "target_id": row["target_id"],
            "route_id": row["route_id"],
            "split": row["split"],
            "base_scenario_config_hash": row["base_scenario_config_hash"],
            "applied_visual_randomization_fields": protocol["randomization"][
                "applied_visual_fields"
            ],
            "randomization_fields_not_applied_to_visuals": protocol[
                "randomization"
            ]["explicitly_not_applied"],
        }
    )
    report["visual_scenario_config_hash"] = object_sha256(
        {
            "applied_visual_randomization_fields": report[
                "applied_visual_randomization_fields"
            ],
            "equipment_changes": report["equipment_changes"],
            "light_intensity": report["light_intensity"],
            "unknown_obstacles": report["unknown_obstacles"],
            "feasibility_adjustments": report.get("feasibility_adjustments", []),
        }
    )
    report["world_sha256"] = file_sha256(world_path)
    report["planner_config_sha256"] = file_sha256(planner_path)
    write_json(report_path, report)
    recording_directory = (
        Path(output_root) / "recordings" / row["recording_id"]
    )
    flight_events_path = recording_directory / "flight_events.jsonl"
    launcher_environment = {
        "MAP_ID": "custom",
        "WORLD_NAME": entry["world_name"],
        "WORLD_SRC": str(world_path.resolve()),
        "PX4_GZ_MODEL_POSE": spawn_pose_text(entry),
        "SIM_MODEL": "x500_research",
        "HEADLESS": "1",
    }
    return {
        "scenario": row,
        "scenario_report": str(report_path),
        "world_path": str(world_path),
        "planner_path": str(planner_path),
        "recording_directory": str(recording_directory),
        "launcher_environment": launcher_environment,
        "launcher_command": ["bash", "scripts/flight/start_px4_substation.sh"],
        "flight_command": [
            "python", "main.py", "task", "run", "fly_round_trip", "--",
            "--obstacle-config", str(planner_path),
            "--visual-mission-events",
            str(flight_events_path),
        ],
        "flight_events_path": str(flight_events_path),
    }


def collection_recording_context(plan, scenario_id, scenario_report):
    row = scenario_by_id(plan, scenario_id)
    report = _read_json(scenario_report)
    expected = {
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "scenario_id": row["scenario_id"],
        "map_id": row["map_id"],
        "target_id": row["target_id"],
        "seed": row["seed"],
        "route_id": row["route_id"],
        "split": row["split"],
        "base_scenario_config_hash": row["base_scenario_config_hash"],
    }
    mismatched = [
        key for key, value in expected.items() if report.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            "scenario report differs from collection plan: "
            + ", ".join(mismatched)
        )
    return {
        "recording_type": "labelled_visual_collection",
        "protocol_id": "visual-multiscenario-png-v1",
        "dataset_role": row["dataset_role"],
        "map_id": row["map_id"],
        "target_id": row["target_id"],
        "seed": row["seed"],
        "route_id": row["route_id"],
        "scenario_id": row["scenario_id"],
        "split": row["split"],
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "base_scenario_config_hash": row["base_scenario_config_hash"],
        "scenario_config_hash": report["visual_scenario_config_hash"],
        "world_sha256": report["world_sha256"],
        "planner_config_sha256": report["planner_config_sha256"],
    }


def _acceptance_protocol(protocol):
    recording = protocol["recording"]
    minimum = float(recording["minimum_phase_duration_s"])
    return {
        "recording": recording,
        "coverage": {
            "required_mission_phases": recording["required_mission_phases"],
            "mission_phase_minimum_duration_s": {
                phase: minimum
                for phase in recording["required_mission_phases"]
            },
            "minimum_no_target_fraction": 0.0,
        },
    }


def validate_collection_recording(
    recording_directory,
    plan,
    *,
    protocol_path=DEFAULT_PROTOCOL,
):
    root = Path(recording_directory)
    protocol = load_collection_protocol(protocol_path)
    for relative in REQUIRED_MANIFESTS:
        if not (root / relative).is_file():
            raise PilotRecordingError(
                f"missing required collection manifest: {relative}"
            )
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    if (
        metadata.get("visual_recording_schema_version") != 2
        or summary.get("visual_recording_schema_version") != 2
        or metadata.get("recording_type") != "labelled_visual_collection"
        or metadata.get("protocol_id") != protocol["protocol_id"]
        or metadata.get("recording_state") != "complete"
    ):
        raise PilotRecordingError("visual collection recording metadata is invalid")
    row = scenario_by_id(plan, metadata.get("scenario_id"))
    expected = {
        "recording_id": row["recording_id"],
        "map_id": row["map_id"],
        "target_id": row["target_id"],
        "seed": row["seed"],
        "route_id": row["route_id"],
        "split": row["split"],
        "dataset_role": row["dataset_role"],
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "base_scenario_config_hash": row["base_scenario_config_hash"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise PilotRecordingError("collection recording identity differs from plan")
    lifecycle = metadata.get("flight_lifecycle")
    if lifecycle is None:
        raise PilotRecordingError(
            "visual collection requires automatic flight lifecycle evidence"
        )
    if (
        lifecycle.get("event_type") != "mission_completed"
        or lifecycle.get("status") != "completed"
        or lifecycle.get("phase") != "landed"
        or lifecycle.get("landing_confirmed") is not True
    ):
        raise PilotRecordingError("automatic flight lifecycle did not complete")
    flight_events_path = root / "flight_events.jsonl"
    identity = _read_json(root / "identity/recording_identity.json")
    if (
        not flight_events_path.is_file()
        or identity.get("flight_events_manifest_sha256")
        != _sha256(flight_events_path)
    ):
        raise PilotRecordingError("flight lifecycle manifest hash mismatch")
    for name in (
        "scenario_config_hash",
        "world_sha256",
        "planner_config_sha256",
    ):
        value = metadata.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise PilotRecordingError(f"collection metadata lacks {name}")
    frames, synchronization, annotations, _, frames_by_id = (
        _load_and_validate_linkage(root, metadata)
    )
    identity_sha256 = _validate_recording_identity(root)
    truth_events = _read_jsonl(root / "truth_events.jsonl")
    if len(truth_events) != summary["synchronization"]["truth_message_count"]:
        raise PilotRecordingError("collection truth event count mismatch")
    failures, phase_durations = pilot_acceptance_failures(
        frames=frames,
        synchronization=synchronization,
        annotations=annotations,
        frames_by_id=frames_by_id,
        metadata=metadata,
        summary=summary,
        protocol=_acceptance_protocol(protocol),
    )
    if failures:
        raise PilotRecordingError("; ".join(failures))
    return {
        "valid": True,
        "accepted": True,
        "scenario_id": row["scenario_id"],
        "recording_id": row["recording_id"],
        "split": row["split"],
        "source_frame_count": len(frames),
        "dataset_frame_count": len(annotations),
        "mission_phase_duration_s": phase_durations,
        "recording_identity_sha256": identity_sha256,
        "scenario_identity_sha256": object_sha256(
            {
                "scenario_id": row["scenario_id"],
                "scenario_config_hash": metadata["scenario_config_hash"],
            }
        ),
    }
