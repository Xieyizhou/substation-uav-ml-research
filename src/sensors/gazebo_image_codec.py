"""Decode Gazebo image messages into canonical PNG camera frames."""

from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
import time

from PIL import Image

from src.sensors.gazebo_visual_transport import (
    GAZEBO_SIM_CLOCK,
    RGB_MESSAGE_TYPE,
    header_sequence,
    message_timestamp,
)
from src.sensors.types import CameraFrame


PIXEL_FORMATS = {
    1: ("mono8", 1),
    3: ("rgb8", 3),
    4: ("rgba8", 4),
    5: ("bgra8", 4),
    8: ("bgr8", 3),
    "L_INT8": ("mono8", 1),
    "RGB_INT8": ("rgb8", 3),
    "RGBA_INT8": ("rgba8", 4),
    "BGRA_INT8": ("bgra8", 4),
    "BGR_INT8": ("bgr8", 3),
}


def _message_bytes(value):
    if not isinstance(value, str):
        raise ValueError("Gazebo image data must be base64 JSON text")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Gazebo image data is not valid base64") from error


def _pixel_description(value):
    key = value.upper() if isinstance(value, str) else value
    try:
        return PIXEL_FORMATS[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported Gazebo pixel format: {value!r}") from error


def _packed_rows(data, width, height, step, channels):
    packed_step = width * channels
    if step < packed_step:
        raise ValueError("Gazebo image step is smaller than packed row size")
    if len(data) != step * height:
        raise ValueError("Gazebo image payload length does not match step * height")
    if step == packed_step:
        return data
    return b"".join(
        data[row * step : row * step + packed_step] for row in range(height)
    )


def _canonical_rgb_image(data, width, height, source_pixel_format):
    mode = {
        "mono8": "L",
        "rgb8": "RGB",
        "bgr8": "RGB",
        "rgba8": "RGBA",
        "bgra8": "RGBA",
    }[source_pixel_format]
    if source_pixel_format == "bgr8":
        data = bytes(
            component
            for offset in range(0, len(data), 3)
            for component in (data[offset + 2], data[offset + 1], data[offset])
        )
    elif source_pixel_format == "bgra8":
        data = bytes(
            component
            for offset in range(0, len(data), 4)
            for component in (
                data[offset + 2], data[offset + 1], data[offset], data[offset + 3]
            )
        )
    return Image.frombytes(mode, (width, height), data).convert("RGB")


def parse_gazebo_image_message(
    message, *, topic, staging_directory, receive_index, received_monotonic=None
):
    width = int(message.get("width", 0))
    height = int(message.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Gazebo image dimensions must be positive")
    pixel_value = message.get("pixel_format_type", message.get("pixelFormatType"))
    source_pixel_format, channels = _pixel_description(pixel_value)
    step = int(message.get("step", width * channels))
    source_bytes = _message_bytes(message.get("data"))
    packed = _packed_rows(source_bytes, width, height, step, channels)
    source_sequence = header_sequence(message)
    sequence = receive_index if source_sequence is None else source_sequence
    provenance = "local_receive_ordinal" if source_sequence is None else "gazebo_header"
    directory = Path(staging_directory)
    (directory / "frames").mkdir(parents=True, exist_ok=True)
    relative_path = f"frames/{receive_index:09d}.png"
    destination = directory / relative_path
    _canonical_rgb_image(packed, width, height, source_pixel_format).save(
        destination, format="PNG", optimize=False, compress_level=6
    )
    payload = destination.read_bytes()
    return CameraFrame(
        frame_id=f"gazebo-rgb-{receive_index:09d}",
        source_id=f"gazebo_rgb:{topic}",
        sequence_number=sequence,
        capture_timestamp=message_timestamp(message),
        capture_clock_domain=GAZEBO_SIM_CLOCK,
        receive_monotonic_timestamp=(
            time.monotonic() if received_monotonic is None else received_monotonic
        ),
        width=width,
        height=height,
        payload_format="png",
        pixel_format="rgb8",
        payload_relative_path=relative_path,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        metadata={
            "runtime_topic": topic,
            "gazebo_message_type": RGB_MESSAGE_TYPE,
            "gazebo_pixel_format": pixel_value,
            "source_pixel_format": source_pixel_format,
            "source_row_stride_bytes": step,
            "source_payload_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "sequence_provenance": provenance,
        },
    )
