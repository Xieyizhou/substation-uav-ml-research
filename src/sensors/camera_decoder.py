"""Deterministic CPU camera payload decoding with honest stage timings."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.sensors.camera_decoded import (
    CameraDecoderError,
    DecodedCameraResult,
    DecodedImage,
    DecoderConfiguration,
    DecoderConfigurationError,
    DecoderDeterminismError,
    DecoderDimensionMismatchError,
    DecoderMalformedPayloadError,
    DecoderPayloadFormatMismatchError,
    DecoderPayloadIntegrityError,
    DecoderPayloadMissingError,
    DecoderRawByteCountError,
    DecoderUnsupportedDtypeError,
    DecoderUnsupportedPixelFormatError,
    DecoderUnsupportedStrideError,
    decoded_content_sha256,
)
from src.sensors.types import (
    CAMERA_PIXEL_CHANNELS,
    CAMERA_RAW_LAYOUT,
    CameraFrame,
    VisualTiming,
)


def _frame_payload_path(frame, recording_root):
    root = Path(recording_root).resolve()
    payload = (root / frame.payload_relative_path).resolve()
    try:
        payload.relative_to(root)
    except ValueError as error:
        raise DecoderPayloadMissingError(
            frame, f"payload path escapes recording root: {frame.payload_relative_path}"
        ) from error
    return payload


def _validate_declared_extension(frame):
    extension = Path(frame.payload_relative_path).suffix.lower()
    allowed = {
        "png": {".png"},
        "jpeg": {".jpg", ".jpeg"},
        "raw": {".raw"},
    }[frame.payload_format]
    if extension not in allowed:
        raise DecoderPayloadFormatMismatchError(
            frame,
            f"declared payload_format={frame.payload_format!r} conflicts with "
            f"extension {extension!r}",
        )


def _validate_raw_contract(frame, recording_root):
    metadata_path = Path(recording_root) / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise DecoderUnsupportedStrideError(
            frame, "raw payload requires valid recording metadata"
        ) from error
    if metadata.get("raw_layout_contract") != CAMERA_RAW_LAYOUT:
        raise DecoderUnsupportedStrideError(
            frame,
            f"raw_layout_contract must be {CAMERA_RAW_LAYOUT!r}",
        )
    dtype = frame.metadata.get("dtype", "uint8")
    if dtype != "uint8":
        raise DecoderUnsupportedDtypeError(
            frame, f"schema v1 raw dtype must be uint8, got {dtype!r}"
        )
    expected_stride = frame.width * CAMERA_PIXEL_CHANNELS[frame.pixel_format]
    row_stride = frame.metadata.get("row_stride_bytes", expected_stride)
    if row_stride != expected_stride:
        raise DecoderUnsupportedStrideError(
            frame,
            f"schema v1 raw row_stride_bytes must be {expected_stride}, "
            f"got {row_stride!r}",
        )
    byte_order = frame.metadata.get("byte_order", "not_applicable")
    if byte_order != "not_applicable":
        raise DecoderUnsupportedDtypeError(
            frame,
            "schema v1 uint8 raw byte_order must be 'not_applicable'",
        )


def _to_canonical_rgb(frame, array):
    if frame.pixel_format == "mono8":
        mono = array if array.ndim == 2 else array[:, :, 0]
        rgb = np.repeat(mono[:, :, np.newaxis], 3, axis=2)
    elif frame.pixel_format == "rgb8":
        rgb = array
    elif frame.pixel_format == "bgr8":
        rgb = array[:, :, [2, 1, 0]]
    elif frame.pixel_format == "rgba8":
        rgb = array[:, :, :3]
    elif frame.pixel_format == "bgra8":
        rgb = array[:, :, [2, 1, 0]]
    else:
        raise DecoderUnsupportedPixelFormatError(
            frame, f"unsupported pixel_format={frame.pixel_format!r}"
        )
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _decode_encoded(frame, payload_bytes):
    expected_backend_format = {"png": "PNG", "jpeg": "JPEG"}[frame.payload_format]
    try:
        with Image.open(BytesIO(payload_bytes)) as image:
            if image.format != expected_backend_format:
                raise DecoderPayloadFormatMismatchError(
                    frame,
                    f"declared payload_format={frame.payload_format!r} but "
                    f"decoded content is {image.format!r}",
                )
            image.load()
            if image.size != (frame.width, frame.height):
                raise DecoderDimensionMismatchError(
                    frame,
                    f"decoded dimensions {image.size[0]}x{image.size[1]} do not "
                    f"match declared {frame.width}x{frame.height}",
                )
            mode = image.mode
            array = np.asarray(image)
    except DecoderPayloadFormatMismatchError:
        raise
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise DecoderMalformedPayloadError(
            frame, f"malformed {frame.payload_format} payload"
        ) from error
    expected_pixel_formats = {
        "L": {"mono8"},
        "RGB": {"rgb8", "bgr8"},
        "RGBA": {"rgba8", "bgra8"},
    }
    if mode not in expected_pixel_formats:
        raise DecoderUnsupportedPixelFormatError(
            frame, f"decoded image mode {mode!r} is unsupported"
        )
    if frame.pixel_format not in expected_pixel_formats[mode]:
        raise DecoderUnsupportedPixelFormatError(
            frame,
            f"declared pixel_format={frame.pixel_format!r} conflicts with "
            f"decoded image mode {mode!r}",
        )
    return _to_canonical_rgb(frame, array)


def _decode_raw(frame, payload_bytes, recording_root):
    _validate_raw_contract(frame, recording_root)
    channels = CAMERA_PIXEL_CHANNELS[frame.pixel_format]
    expected_bytes = frame.width * frame.height * channels
    if len(payload_bytes) != expected_bytes:
        raise DecoderRawByteCountError(
            frame,
            f"raw payload has {len(payload_bytes)} byte(s); expected {expected_bytes}",
        )
    array = np.frombuffer(payload_bytes, dtype=np.uint8).reshape(
        frame.height, frame.width, channels
    )
    return _to_canonical_rgb(frame, array)


def decode_camera_payload(frame, recording_root, configuration=None):
    """Load, verify, and decode one payload to canonical RGB8 HWC uint8."""
    if not isinstance(frame, CameraFrame):
        raise TypeError("frame must be a CameraFrame")
    configuration = configuration or DecoderConfiguration()
    if not isinstance(configuration, DecoderConfiguration):
        raise DecoderConfigurationError(
            "configuration must be a DecoderConfiguration"
        )
    _validate_declared_extension(frame)
    payload_path = _frame_payload_path(frame, recording_root)
    load_started_ns = time.monotonic_ns()
    try:
        payload_bytes = payload_path.read_bytes()
    except FileNotFoundError as error:
        raise DecoderPayloadMissingError(
            frame, f"missing payload {frame.payload_relative_path!r}"
        ) from error
    load_finished_ns = time.monotonic_ns()
    decode_started_ns = load_finished_ns
    source_hash = hashlib.sha256(payload_bytes).hexdigest()
    if source_hash != frame.payload_sha256:
        raise DecoderPayloadIntegrityError(
            frame, f"payload SHA256 mismatch for {frame.payload_relative_path!r}"
        )
    if frame.payload_format in {"png", "jpeg"}:
        canonical = _decode_encoded(frame, payload_bytes)
    else:
        canonical = _decode_raw(frame, payload_bytes, recording_root)
    decoded_hash = decoded_content_sha256(canonical)
    image = DecodedImage(
        frame_id=frame.frame_id,
        source_id=frame.source_id,
        sequence_number=frame.sequence_number,
        width=frame.width,
        height=frame.height,
        pixel_format="rgb8",
        dtype="uint8",
        layout="HWC",
        source_payload_sha256=source_hash,
        decoded_content_sha256=decoded_hash,
        decoder_id=configuration.decoder_id,
        decoder_version=configuration.decoder_version,
        decoder_configuration_id=configuration.configuration_id,
        array=canonical,
    )
    decode_finished_ns = time.monotonic_ns()
    timing = VisualTiming(
        payload_load_ms=(load_finished_ns - load_started_ns) / 1_000_000,
        decode_ms=(decode_finished_ns - decode_started_ns) / 1_000_000,
        input_width=frame.width,
        input_height=frame.height,
        timing_provenance={
            "payload_load_ms": "local_monotonic",
            "decode_ms": "local_monotonic",
        },
    )
    return DecodedCameraResult(image=image, timing=timing)


def validate_decode_determinism(
    frames,
    recording_root,
    *,
    configuration=None,
    repetitions=2,
):
    """Compare canonical identities while deliberately ignoring timing values."""
    if repetitions < 2:
        raise ValueError("determinism validation requires at least two repetitions")
    frames = tuple(frames)
    configuration = configuration or DecoderConfiguration()
    baseline = None
    for _ in range(repetitions):
        identities = tuple(
            (
                result.image.frame_id,
                result.image.sequence_number,
                result.image.width,
                result.image.height,
                result.image.pixel_format,
                result.image.dtype,
                result.image.layout,
                result.image.source_payload_sha256,
                result.image.decoded_content_sha256,
                result.image.decoder_configuration_id,
            )
            for result in (
                decode_camera_payload(frame, recording_root, configuration)
                for frame in frames
            )
        )
        if baseline is None:
            baseline = identities
        elif identities != baseline:
            mismatch_frame = frames[0] if frames else ("recording", 0)
            raise DecoderDeterminismError(
                mismatch_frame, "repeated canonical decode identities differ"
            )
    return {
        "deterministic": True,
        "frame_count": len(frames),
        "repetitions": repetitions,
        "decoder_configuration_id": configuration.configuration_id,
        "decoded_content_sha256": [identity[8] for identity in baseline],
    }
