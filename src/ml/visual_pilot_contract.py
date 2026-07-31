"""Versioned pilot policy and recording-context helpers."""

from __future__ import annotations

from pathlib import Path

from src.ml.visual_pilot_acceptance import validate_v3_acceptance_policy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "benchmarks/visual_static_v1/pilot_protocol.json"


def load_pilot_protocol(path=DEFAULT_PROTOCOL):
    from src.ml.visual_pilot import PilotRecordingError, _read_json

    protocol = _read_json(path)
    if protocol.get("protocol_id") != "visual-pilot-png-v3":
        raise PilotRecordingError("pilot protocol must be visual-pilot-png-v3")
    recording = protocol.get("recording") or {}
    if not recording.get("retain_every_valid_source_frame"):
        raise PilotRecordingError("pilot protocol must retain every valid frame")
    if recording.get("automatic_downsampling"):
        raise PilotRecordingError("pilot protocol cannot downsample automatically")
    try:
        validate_v3_acceptance_policy(protocol)
    except ValueError as error:
        raise PilotRecordingError(str(error)) from error
    return protocol


def recording_context(protocol, override=None):
    scenario = protocol["scenario"]
    default = {
        "recording_type": "labelled_visual_pilot",
        "protocol_id": protocol["protocol_id"],
        "dataset_role": "pilot",
        "map_id": scenario["map_id"],
        "target_id": scenario["target_id"],
        "seed": int(scenario["seed"]),
        "route_id": scenario["route"],
        "scenario_id": (
            f"{scenario['map_id']}-{scenario['target_id']}-{scenario['seed']}"
        ),
        "split": None,
        "collection_plan_identity_sha256": None,
        "base_scenario_config_hash": None,
        "scenario_config_hash": None,
    }
    value = default if override is None else dict(override)
    required = {
        "recording_type",
        "protocol_id",
        "dataset_role",
        "map_id",
        "target_id",
        "seed",
        "route_id",
        "scenario_id",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"visual recording context is missing fields: {missing}")
    if value["recording_type"] not in {
        "labelled_visual_pilot",
        "labelled_visual_collection",
    }:
        raise ValueError("unsupported visual recording type")
    return value


def mission_phase(frame, events):
    applicable = [
        event
        for event in events
        if float(event["simulation_timestamp"]) <= frame.capture_timestamp
    ]
    return applicable[-1]["mission_phase"] if applicable else "other"


def validate_mission_events(events):
    from src.ml.visual_pilot import PilotRecordingError

    previous = None
    allowed = {
        "cruise_distant",
        "approach",
        "close_inspection",
        "target_transition",
        "other",
    }
    normalized = []
    for event in events:
        timestamp = float(event["simulation_timestamp"])
        phase = str(event["mission_phase"])
        if timestamp < 0 or (previous is not None and timestamp < previous):
            raise PilotRecordingError("mission events must use ordered timestamps")
        if phase not in allowed:
            raise PilotRecordingError(f"unsupported mission phase: {phase}")
        record = {"simulation_timestamp": timestamp, "mission_phase": phase}
        for name in (
            "event_source",
            "flight_event_sequence",
            "flight_event_type",
            "flight_phase",
        ):
            if name in event:
                record[name] = event[name]
        normalized.append(record)
        previous = timestamp
    return tuple(normalized)
