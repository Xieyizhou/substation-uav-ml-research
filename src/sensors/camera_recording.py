"""Portable, deterministic camera recording and replay."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

from src.sensors.types import (
    CAMERA_PIXEL_CHANNELS,
    CAMERA_RAW_LAYOUT,
    CameraFrame,
)


CAMERA_RECORDING_SCHEMA_VERSION = 1


class CameraRecordingError(ValueError):
    """Base class for invalid camera recording artifacts."""


class UnsupportedCameraRecordingSchema(CameraRecordingError):
    pass


class CameraManifestError(CameraRecordingError):
    pass


class CameraPayloadMissingError(CameraRecordingError):
    pass


class CameraPayloadCorruptedError(CameraRecordingError):
    pass


class CameraPayloadLayoutError(CameraPayloadCorruptedError):
    pass


class CameraReplayTimestampError(CameraRecordingError):
    pass


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _payload_extension(frame):
    if frame.payload_format == "png":
        return ".png"
    if frame.payload_format == "jpeg":
        return ".jpg"
    return ".raw"


def _validate_payload_layout(frame, payload):
    if frame.payload_format != "raw":
        return
    expected_bytes = (
        frame.width * frame.height * CAMERA_PIXEL_CHANNELS[frame.pixel_format]
    )
    actual_bytes = Path(payload).stat().st_size
    if actual_bytes != expected_bytes:
        raise CameraPayloadLayoutError(
            f"{frame.payload_relative_path}: raw payload has {actual_bytes} byte(s); "
            f"expected {expected_bytes} for {frame.width}x{frame.height} "
            f"{frame.pixel_format} under {CAMERA_RAW_LAYOUT}"
        )


def _recording_summary(frames, payload_bytes, invalid_frames):
    sequences = [frame.sequence_number for frame in frames]
    duplicate_sequence_count = len(sequences) - len(set(sequences))
    missing_sequence_count = 0
    if sequences:
        missing_sequence_count = (
            max(sequences) - min(sequences) + 1 - len(set(sequences))
        )
        missing_sequence_count = max(0, missing_sequence_count)
    non_monotonic_timestamp_count = 0
    for previous, current in zip(frames, frames[1:]):
        if (
            previous.capture_clock_domain == current.capture_clock_domain
            and current.capture_timestamp < previous.capture_timestamp
        ):
            non_monotonic_timestamp_count += 1
    one_clock_domain = len({frame.capture_clock_domain for frame in frames}) == 1
    timestamp_span_s = None
    if (
        len(frames) > 1
        and one_clock_domain
        and non_monotonic_timestamp_count == 0
    ):
        timestamp_span_s = (
            frames[-1].capture_timestamp - frames[0].capture_timestamp
        )
    effective_frequency_hz = None
    if timestamp_span_s is not None and timestamp_span_s > 0:
        effective_frequency_hz = (len(frames) - 1) / timestamp_span_s
    return {
        "recording_schema_version": CAMERA_RECORDING_SCHEMA_VERSION,
        "accepted_frame_count": len(frames),
        "invalid_frame_count": len(invalid_frames),
        "invalid_reason_counts": dict(
            sorted(Counter(item["reason"] for item in invalid_frames).items())
        ),
        "missing_sequence_count": missing_sequence_count,
        "duplicate_sequence_count": duplicate_sequence_count,
        "non_monotonic_timestamp_count": non_monotonic_timestamp_count,
        "timestamp_span_s": timestamp_span_s,
        "effective_frequency_hz": effective_frequency_hz,
        "total_payload_bytes": sum(payload_bytes),
        "first_sequence_number": sequences[0] if sequences else None,
        "last_sequence_number": sequences[-1] if sequences else None,
        "first_frame_id": frames[0].frame_id if frames else None,
        "last_frame_id": frames[-1].frame_id if frames else None,
        "capture_clock_domains": sorted(
            {frame.capture_clock_domain for frame in frames}
        ),
        "payload_formats": sorted({frame.payload_format for frame in frames}),
        "pixel_formats": sorted({frame.pixel_format for frame in frames}),
        "raw_layout_contract": CAMERA_RAW_LAYOUT,
        "hash_validation_status": "passed",
        "timestamp_health_status": (
            "healthy"
            if frames and one_clock_domain and not non_monotonic_timestamp_count
            else "mixed_or_non_monotonic"
            if frames
            else "unavailable"
        ),
    }


class _CameraRecordingWriter:
    def __init__(self, output_directory, payload_root=None):
        self.output_directory = Path(output_directory)
        self.payload_root = Path(payload_root or ".")
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise CameraRecordingError(
                f"camera recording directory is not empty: {self.output_directory}"
            )
        self.frames_directory = self.output_directory / "frames"
        self.frames_directory.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.payload_bytes = []
        self.invalid_frames = []

    def append(self, item, input_index):
        try:
            frame = item if isinstance(item, CameraFrame) else CameraFrame.from_record(item)
            source_payload = self.payload_root / frame.payload_relative_path
            if not source_payload.is_file():
                raise CameraPayloadMissingError(
                    f"missing payload: {frame.payload_relative_path}"
                )
            _validate_payload_layout(frame, source_payload)
            actual_sha256 = _file_sha256(source_payload)
            if actual_sha256 != frame.payload_sha256:
                raise CameraPayloadCorruptedError(
                    f"payload hash mismatch: {frame.payload_relative_path}"
                )
            relative_payload = (
                f"frames/{len(self.frames):09d}{_payload_extension(frame)}"
            )
            destination = self.output_directory / relative_payload
            shutil.copyfile(source_payload, destination)
            stored_frame = replace(
                frame,
                payload_relative_path=relative_payload,
                payload_sha256=_file_sha256(destination),
            )
            self.frames.append(stored_frame)
            self.payload_bytes.append(destination.stat().st_size)
        except (CameraRecordingError, OSError, TypeError, ValueError) as error:
            self.invalid_frames.append(
                {
                    "input_index": input_index,
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )

    def finish(self):
        frames_path = self.output_directory / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as output:
            for frame in self.frames:
                output.write(
                    json.dumps(
                        frame.to_record(),
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        summary = _recording_summary(
            self.frames, self.payload_bytes, self.invalid_frames
        )
        metadata = {
            "recording_schema_version": CAMERA_RECORDING_SCHEMA_VERSION,
            "recording_type": "camera_frame_collection",
            "manifest": "frames.jsonl",
            "payload_directory": "frames",
            "manifest_order_authoritative": True,
            "source_ids": sorted({frame.source_id for frame in self.frames}),
        "capture_clock_domains": summary["capture_clock_domains"],
        "payload_formats": summary["payload_formats"],
        "pixel_formats": summary["pixel_formats"],
        "raw_layout_contract": CAMERA_RAW_LAYOUT,
        "invalid_frames": self.invalid_frames,
        }
        _write_json(self.output_directory / "metadata.json", metadata)
        _write_json(self.output_directory / "summary.json", summary)
        return summary


def record_camera_frames(frames, output_directory, *, payload_root=None):
    """Record a synchronous iterable without embedding payload bytes in JSONL."""
    writer = _CameraRecordingWriter(output_directory, payload_root)
    for input_index, item in enumerate(frames):
        writer.append(item, input_index)
    return writer.finish()


async def record_camera_frames_async(frames, output_directory, *, payload_root=None):
    """Record an async iterable using the same deterministic layout."""
    writer = _CameraRecordingWriter(output_directory, payload_root)
    input_index = 0
    async for item in frames:
        writer.append(item, input_index)
        input_index += 1
    return writer.finish()


def _read_metadata(recording_directory):
    path = Path(recording_directory) / "metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CameraManifestError(f"missing recording metadata: {path}") from error
    except json.JSONDecodeError as error:
        raise CameraManifestError(f"malformed recording metadata: {error}") from error
    version = metadata.get("recording_schema_version")
    if version != CAMERA_RECORDING_SCHEMA_VERSION:
        raise UnsupportedCameraRecordingSchema(
            f"unsupported camera recording schema version: {version!r}"
        )
    if metadata.get("recording_type") != "camera_frame_collection":
        raise CameraManifestError("unsupported camera recording type")
    return metadata


def _read_manifest(recording_directory):
    recording_directory = Path(recording_directory)
    frames_path = recording_directory / "frames.jsonl"
    frames = []
    try:
        source = frames_path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise CameraManifestError(f"missing camera manifest: {frames_path}") from error
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frames.append(CameraFrame.from_record(record))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise CameraManifestError(
                    f"{frames_path}:{line_number}: {error}"
                ) from error
    return frames


def validate_camera_recording(recording_directory):
    """Verify schema, manifest rows, payload existence, and every payload hash."""
    recording_directory = Path(recording_directory)
    metadata = _read_metadata(recording_directory)
    frames = _read_manifest(recording_directory)
    payload_bytes = []
    for frame in frames:
        payload = recording_directory / frame.payload_relative_path
        if not payload.is_file():
            raise CameraPayloadMissingError(
                f"missing camera payload: {frame.payload_relative_path}"
            )
        _validate_payload_layout(frame, payload)
        actual_sha256 = _file_sha256(payload)
        if actual_sha256 != frame.payload_sha256:
            raise CameraPayloadCorruptedError(
                f"camera payload hash mismatch: {frame.payload_relative_path}"
            )
        payload_bytes.append(payload.stat().st_size)
    invalid_frames = metadata.get("invalid_frames", [])
    if not isinstance(invalid_frames, list):
        raise CameraManifestError("metadata invalid_frames must be a list")
    summary = _recording_summary(frames, payload_bytes, invalid_frames)
    summary["recording_directory"] = str(recording_directory)
    return summary


class CameraReplaySource:
    """Manifest-ordered camera replay with verified external payloads."""

    def __init__(self, recording_directory):
        self.recording_directory = Path(recording_directory)
        self._frames = None
        self._summary = None

    def load(self):
        self._summary = validate_camera_recording(self.recording_directory)
        self._frames = _read_manifest(self.recording_directory)
        return self._frames

    @property
    def summary(self):
        if self._summary is None:
            self.load()
        return dict(self._summary)

    def payload_path(self, frame):
        return self.recording_directory / frame.payload_relative_path

    def iter_frames(self):
        """Yield verified frames in manifest order without sleeping."""
        for frame in self.load():
            yield frame

    async def replay(self, *, mode="no_sleep", rate=1.0):
        if mode not in {"no_sleep", "paced"}:
            raise ValueError("camera replay mode must be 'no_sleep' or 'paced'")
        if rate <= 0:
            raise ValueError("camera replay rate must be positive")
        previous = None
        for frame in self.load():
            if mode == "paced" and previous is not None:
                if frame.capture_clock_domain != previous.capture_clock_domain:
                    raise CameraReplayTimestampError(
                        "paced replay requires one capture clock domain"
                    )
                delay = frame.capture_timestamp - previous.capture_timestamp
                if delay < 0:
                    raise CameraReplayTimestampError(
                        "paced replay requires monotonic capture timestamps"
                    )
                await asyncio.sleep(delay / rate)
            yield frame
            previous = frame

    def replay_summary(self, *, mode="no_sleep"):
        summary = self.summary
        summary["replay_mode"] = mode
        summary["ordering_rule"] = "frames.jsonl manifest order"
        return summary
