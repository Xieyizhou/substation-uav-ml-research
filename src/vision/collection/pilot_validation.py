"""Validation and pilot-only dataset identity materialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ml.artifacts import object_sha256
from src.vision.contracts.annotations import VisualFrameAnnotation
from src.vision.contracts.identity import DatasetIdentity, class_order_identity
from src.vision.collection.pilot import (
    REQUIRED_MANIFESTS,
    PilotRecordingError,
    _read_json,
    _read_jsonl,
    _sha256,
    _validate_mission_events,
    _write_json,
    _write_jsonl,
    load_pilot_protocol,
)
from src.vision.collection.pilot_acceptance import pilot_acceptance_failures
from src.vision.collection.synchronization import SynchronizationRecord
from src.sensors.camera_decoder import decode_camera_payload
from src.sensors.types import CameraFrame


def inspect_pilot_recording(recording_directory):
    root = Path(recording_directory)
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    return {
        "recording_directory": str(root),
        "recording_id": metadata.get("recording_id"),
        "recording_type": metadata.get("recording_type"),
        "recording_state": metadata.get("recording_state"),
        "protocol_id": metadata.get("protocol_id"),
        "source_health": summary.get("source_health"),
        "synchronization": summary.get("synchronization"),
        "dataset_identity_exists": (root / "identity/dataset_identity.json").is_file(),
    }


def _validate_payload(root, frame):
    payload = root / frame.payload_relative_path
    if not payload.is_file() or _sha256(payload) != frame.payload_sha256:
        raise PilotRecordingError(f"invalid pilot payload: {frame.frame_id}")


def _validate_annotation(root, metadata, frames_by_id, annotation):
    frame = frames_by_id.get(annotation.frame_id)
    if frame is None:
        raise PilotRecordingError("annotation references an unknown RGB frame")
    annotation.validate_linkage(
        frame, decode_camera_payload(frame, root).image
    )
    if annotation.recording_id != metadata["recording_id"]:
        raise PilotRecordingError("annotation recording identity mismatch")


def _parallel_validate(items, function):
    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(function, items))


def _load_and_validate_linkage(root, metadata):
    frames = [
        CameraFrame.from_record(row) for row in _read_jsonl(root / "frames.jsonl")
    ]
    synchronization = [
        SynchronizationRecord(**row)
        for row in _read_jsonl(root / "synchronization.jsonl")
    ]
    annotations = [
        VisualFrameAnnotation.from_record(row)
        for row in _read_jsonl(root / "annotations.jsonl")
    ]
    events = _validate_mission_events(_read_jsonl(root / "mission_events.jsonl"))
    if len(frames) != len(synchronization):
        raise PilotRecordingError("every RGB frame requires synchronization status")
    for frame, sync in zip(frames, synchronization):
        if (
            sync.rgb_frame_id != frame.frame_id
            or sync.rgb_sequence_number != frame.sequence_number
            or sync.rgb_simulation_timestamp != frame.capture_timestamp
        ):
            raise PilotRecordingError("synchronization manifest is not frame-ordered")
    _parallel_validate(frames, lambda frame: _validate_payload(root, frame))
    frames_by_id = {frame.frame_id: frame for frame in frames}
    if len(frames_by_id) != len(frames):
        raise PilotRecordingError("pilot frame IDs must be unique")
    _parallel_validate(
        annotations,
        lambda annotation: _validate_annotation(
            root, metadata, frames_by_id, annotation
        ),
    )
    annotation_ids = [annotation.frame_id for annotation in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise PilotRecordingError("pilot has duplicate annotation frame linkage")
    return frames, synchronization, annotations, events, frames_by_id


def _validate_recording_identity(root):
    identity = _read_json(root / "identity/recording_identity.json")
    if identity.get("recording_identity_schema_version") != 2:
        raise PilotRecordingError("unsupported recording identity schema")
    supplied = identity.pop("recording_identity_sha256", None)
    if supplied != object_sha256(identity):
        raise PilotRecordingError("recording identity SHA256 mismatch")
    expected = {
        "frames_manifest_sha256": _sha256(root / "frames.jsonl"),
        "annotation_manifest_sha256": _sha256(root / "annotations.jsonl"),
        "synchronization_manifest_sha256": _sha256(
            root / "synchronization.jsonl"
        ),
        "truth_events_manifest_sha256": _sha256(root / "truth_events.jsonl"),
        "mission_events_manifest_sha256": _sha256(
            root / "mission_events.jsonl"
        ),
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        raise PilotRecordingError("recording identity manifest hash mismatch")
    return supplied


def validate_pilot_recording(recording_directory, *, require_acceptance=True):
    root = Path(recording_directory)
    for relative in REQUIRED_MANIFESTS:
        if not (root / relative).is_file():
            raise PilotRecordingError(f"missing required pilot manifest: {relative}")
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    if (
        metadata.get("pilot_recording_schema_version") != 2
        or summary.get("pilot_recording_schema_version") != 2
    ):
        raise PilotRecordingError("pilot v3 requires recording schema version 2")
    protocol = load_pilot_protocol()
    if metadata.get("protocol_id") != protocol["protocol_id"]:
        raise PilotRecordingError(
            "pilot recording protocol does not match the active reviewed protocol"
        )
    if metadata.get("recording_type") != "labelled_visual_pilot":
        raise PilotRecordingError("camera-only recording is not a labelled pilot")
    if metadata.get("recording_state") != "complete":
        raise PilotRecordingError("incomplete pilot recording cannot pass validation")
    frames, synchronization, annotations, events, frames_by_id = (
        _load_and_validate_linkage(root, metadata)
    )
    supplied_identity = _validate_recording_identity(root)
    truth_events = _read_jsonl(root / "truth_events.jsonl")
    invalid_truth_events = [
        event
        for event in truth_events
        if event.get("validation_status") == "invalid"
    ]
    if any(not event.get("invalid_reasons") for event in invalid_truth_events):
        raise PilotRecordingError("invalid truth event is missing diagnostic reasons")
    expected_truth_count = (summary.get("synchronization") or {}).get(
        "truth_message_count"
    )
    if expected_truth_count != len(truth_events):
        raise PilotRecordingError("truth event count does not match summary")
    synchronization_summary = summary.get("synchronization") or {}
    if synchronization_summary.get("invalid_truth_count") != len(
        invalid_truth_events
    ):
        raise PilotRecordingError("invalid truth count does not match summary")
    if require_acceptance:
        failures, phase_durations = pilot_acceptance_failures(
            frames=frames,
            synchronization=synchronization,
            annotations=annotations,
            frames_by_id=frames_by_id,
            metadata=metadata,
            summary=summary,
            protocol=protocol,
        )
        if failures:
            raise PilotRecordingError("; ".join(failures))
    else:
        phase_durations = {}
    statuses = [item.synchronization_status for item in synchronization]
    return {
        "valid": True,
        "accepted": bool(require_acceptance),
        "recording_id": metadata["recording_id"],
        "source_frame_count": len(frames),
        "frame_count": len(annotations),
        "annotation_count": len(annotations),
        "excluded_non_dataset_frame_count": len(frames) - len(annotations),
        "excluded_invalid_truth_frame_count": statuses.count("invalid_truth"),
        "mission_phase_duration_s": phase_durations,
        "recording_identity_sha256": supplied_identity,
    }


def materialize_pilot_dataset_identity(recording_directory):
    root = Path(recording_directory)
    validation = validate_pilot_recording(root, require_acceptance=True)
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    annotations = _read_jsonl(root / "annotations.jsonl")
    membership = [
        {
            "frame_id": item["frame_id"],
            "sequence_number": item["sequence_number"],
            "payload_sha256": item["payload_sha256"],
            "frame_order_reference": item["frame_order_reference"],
        }
        for item in annotations
    ]
    membership_path = root / "identity/dataset_membership.jsonl"
    _write_jsonl(membership_path, membership)
    membership_sha256 = _sha256(membership_path)
    scenario_record = {
        "map_id": metadata["map_id"],
        "target_id": metadata["target_id"],
        "seed": metadata["seed"],
        "route_id": metadata["route_id"],
    }
    split_record = {
        "dataset_role": "pilot",
        "minimum_split_unit": "recording",
        "recording_id": metadata["recording_id"],
        "included_frame_manifest_sha256": membership_sha256,
    }
    identity = DatasetIdentity(
        dataset_name=metadata["recording_id"],
        dataset_version=metadata["protocol_id"],
        dataset_role="pilot",
        recording_schema_version=2,
        annotation_schema_version=1,
        decoder_configuration_id=summary["decoder_configuration_id"],
        recording_manifest_sha256=membership_sha256,
        scenario_manifest_sha256=object_sha256(scenario_record),
        split_manifest_sha256=object_sha256(split_record),
        ordered_frame_count=validation["frame_count"],
        labelled_frame_count=sum(
            item.get("annotation_status") == "labelled" for item in annotations
        ),
        source_recording_ids=(metadata["recording_id"],),
        scenario_ids=(f"training-center-{metadata['seed']}",),
        map_ids=(metadata["map_id"],),
        seed_ids=(metadata["seed"],),
        class_order_identity=class_order_identity(),
        annotation_manifest_sha256=_sha256(root / "annotations.jsonl"),
        source_payload_formats=("png",),
    )
    _write_json(root / "identity/dataset_identity.json", identity.to_record())
    return identity
