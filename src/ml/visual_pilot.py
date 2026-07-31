"""Pilot recording, validation, inspection, and identity materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from src.ml.artifacts import object_sha256
from src.ml.visual_pilot_metrics import (
    summarize_source_health,
    summarize_synchronization,
)
from src.ml.visual_pilot_contract import (
    DEFAULT_PROTOCOL,
    load_pilot_protocol,
    mission_phase as _mission_phase,
    recording_context,
    validate_mission_events as _validate_mission_events,
)
from src.ml.visual_synchronization import (
    DEFAULT_MAX_SKEW_MS,
    materialize_frame_annotation,
    synchronize_visual_frames,
)
from src.sensors.camera_decoder import decode_camera_payload
from src.sensors.types import CameraFrame


REQUIRED_MANIFESTS = (
    "metadata.json",
    "frames.jsonl",
    "annotations.jsonl",
    "synchronization.jsonl",
    "truth_events.jsonl",
    "mission_events.jsonl",
    "summary.json",
    "identity/recording_identity.json",
)


class PilotRecordingError(ValueError):
    """Pilot recording or validation is incomplete or inconsistent."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(
                json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
            )


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PilotRecordingError(f"missing pilot file: {path}") from error
    except json.JSONDecodeError as error:
        raise PilotRecordingError(f"malformed pilot JSON: {path}: {error}") from error


def _read_jsonl(path):
    records = []
    try:
        source = Path(path).open(encoding="utf-8")
    except FileNotFoundError as error:
        raise PilotRecordingError(f"missing pilot file: {path}") from error
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise PilotRecordingError(
                    f"{path}:{line_number}: malformed JSON: {error}"
                ) from error
    return records


def _copy_payloads(frames, source_root, output):
    copied = []
    for frame in frames:
        source = Path(source_root) / frame.payload_relative_path
        destination = Path(output) / frame.payload_relative_path
        if source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if _sha256(destination) != frame.payload_sha256:
            raise PilotRecordingError(
                f"payload hash mismatch for {frame.payload_relative_path}"
            )
        copied.append(destination.stat().st_size)
    return copied


def write_pilot_recording(
    output_directory,
    frames,
    truths,
    *,
    recording_id,
    mission_events,
    payload_root=None,
    invalid_frames=(),
    source_timeout_count=0,
    complete=True,
    maximum_skew_ms=DEFAULT_MAX_SKEW_MS,
    expected_rate_tolerance_fraction=None,
    jitter_p95_limit_ms=None,
    receive_stall_limit_ms=None,
    protocol_path=DEFAULT_PROTOCOL,
    recording_context_override=None,
):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frames = tuple(
        frame if isinstance(frame, CameraFrame) else CameraFrame.from_record(frame)
        for frame in frames
    )
    truths = tuple(truths)
    invalid_frames = tuple(invalid_frames)
    events = _validate_mission_events(mission_events)
    protocol = load_pilot_protocol(protocol_path)
    if expected_rate_tolerance_fraction is None:
        expected_rate_tolerance_fraction = protocol["recording"][
            "default_rate_observation_tolerance_fraction"
        ]
    context = recording_context(protocol, recording_context_override)
    metadata = {
        "visual_recording_schema_version": 2,
        "recording_type": context["recording_type"],
        "recording_id": recording_id,
        "recording_state": "in_progress",
        **context,
        "canonical_payload_format": "png",
        "canonical_pixel_format": "rgb8",
        "invalid_frames": list(invalid_frames),
    }
    if context["recording_type"] == "labelled_visual_pilot":
        metadata["pilot_recording_schema_version"] = 2
    _write_json(output / "metadata.json", metadata)
    payload_sizes = _copy_payloads(frames, payload_root or output, output)
    _write_jsonl(output / "frames.jsonl", [frame.to_record() for frame in frames])
    _write_jsonl(output / "mission_events.jsonl", events)
    _write_jsonl(output / "truth_events.jsonl", [truth.to_record() for truth in truths])
    synchronized = synchronize_visual_frames(
        frames, truths, maximum_skew_ms=maximum_skew_ms
    )
    annotations = []
    decoder_configuration_id = None
    for index, item in enumerate(synchronized):
        if item.truth is None:
            annotations.append(None)
            continue
        decoded = decode_camera_payload(item.frame, output)
        decoded_image = decoded.image
        if (
            decoder_configuration_id is not None
            and decoded_image.decoder_configuration_id != decoder_configuration_id
        ):
            raise PilotRecordingError("pilot frames used multiple decoder identities")
        decoder_configuration_id = decoded_image.decoder_configuration_id
        annotations.append(
            materialize_frame_annotation(
                item,
                decoded_image,
                recording_id=recording_id,
                scenario_id=context["scenario_id"],
                map_id=context["map_id"],
                seed=context["seed"],
                mission_phase=_mission_phase(item.frame, events),
                frame_order_reference=index,
            )
        )
    annotation_records = [
        annotation.to_record() for annotation in annotations if annotation is not None
    ]
    _write_jsonl(output / "annotations.jsonl", annotation_records)
    _write_jsonl(
        output / "synchronization.jsonl",
        [item.synchronization.to_record() for item in synchronized],
    )
    source_summary = summarize_source_health(
        frames,
        expected_source_rate_hz=protocol["recording"]["expected_source_rate_hz"],
        invalid_frame_count=len(invalid_frames),
        source_timeout_count=source_timeout_count,
        total_png_bytes=sum(payload_sizes),
        expected_rate_tolerance_fraction=expected_rate_tolerance_fraction,
        jitter_p95_limit_ms=jitter_p95_limit_ms,
        receive_stall_limit_ms=receive_stall_limit_ms,
    )
    synchronization_summary = summarize_synchronization(
        synchronized, truths, annotations
    )
    summary = {
        "visual_recording_schema_version": 2,
        "recording_id": recording_id,
        "recording_state": "complete" if complete else "in_progress",
        "recording_is_formal_evidence": False,
        "pilot_is_formal_evidence": False,
        "decoder_configuration_id": decoder_configuration_id,
        "source_health": source_summary,
        "synchronization": synchronization_summary,
    }
    if context["recording_type"] == "labelled_visual_pilot":
        summary["pilot_recording_schema_version"] = 2
    _write_json(output / "summary.json", summary)
    identity_record = {
        "recording_identity_schema_version": 2,
        "recording_id": recording_id,
        "protocol_id": context["protocol_id"],
        "recording_state": summary["recording_state"],
        "frames_manifest_sha256": _sha256(output / "frames.jsonl"),
        "annotation_manifest_sha256": _sha256(output / "annotations.jsonl"),
        "synchronization_manifest_sha256": _sha256(
            output / "synchronization.jsonl"
        ),
        "truth_events_manifest_sha256": _sha256(output / "truth_events.jsonl"),
        "mission_events_manifest_sha256": _sha256(
            output / "mission_events.jsonl"
        ),
    }
    flight_events_path = output / "flight_events.jsonl"
    if flight_events_path.is_file():
        identity_record["flight_events_manifest_sha256"] = _sha256(
            flight_events_path
        )
    identity_record["recording_identity_sha256"] = object_sha256(identity_record)
    _write_json(output / "identity/recording_identity.json", identity_record)
    metadata["recording_state"] = summary["recording_state"]
    _write_json(output / "metadata.json", metadata)
    return summary
