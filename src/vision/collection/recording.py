"""Scenario preparation and validation for multi-recording visual datasets."""

from __future__ import annotations

from pathlib import Path

from src.ml.artifacts import object_sha256
from src.vision.collection.plan import (
    load_collection_protocol,
)
from src.vision.contracts.protocol import V2_PROTOCOL_ID, protocol_for_plan
from src.vision.collection.pilot import (
    PilotRecordingError,
    REQUIRED_MANIFESTS,
    _read_json,
    _read_jsonl,
    _sha256,
)
from src.vision.collection.pilot_acceptance import pilot_acceptance_failures
from src.vision.collection.pilot_validation import (
    _load_and_validate_linkage,
    _validate_recording_identity,
)


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
    protocol_path=None,
):
    protocol = (
        protocol_for_plan(plan)
        if protocol_path is None
        else load_collection_protocol(protocol_path)
    )
    row = scenario_by_id(plan, scenario_id)
    if protocol.protocol_id == V2_PROTOCOL_ID:
        from src.vision.collection.v2_recording import prepare_v2_collection_scenario

        return prepare_v2_collection_scenario(plan, row, output_root, protocol)
    from src.vision.collection.v1_recording import prepare_v1_collection_scenario

    return prepare_v1_collection_scenario(plan, row, output_root, protocol)


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
    context = {
        "recording_type": "labelled_visual_collection",
        "protocol_id": plan["protocol_id"],
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
    for name in (
        "layout_id",
        "layout_identity_sha256",
        "route_identity_sha256",
        "route_manifest_sha256",
        "camera_model_sha256",
    ):
        if name in report:
            context[name] = report[name]
    return context


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
    protocol_path=None,
):
    root = Path(recording_directory)
    protocol = (
        protocol_for_plan(plan)
        if protocol_path is None
        else load_collection_protocol(protocol_path)
    )
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
    route_quality = None
    if protocol.protocol_id == V2_PROTOCOL_ID:
        from src.vision.collection.quality import route_quality_failures

        route_quality, quality_failures = route_quality_failures(
            annotations,
            frames_by_id,
            row,
            protocol,
        )
        failures.extend(quality_failures)
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
        "route_quality": route_quality,
        "recording_identity_sha256": identity_sha256,
        "scenario_identity_sha256": object_sha256(
            {
                "scenario_id": row["scenario_id"],
                "scenario_config_hash": metadata["scenario_config_hash"],
            }
        ),
    }
