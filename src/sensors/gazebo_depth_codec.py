"""Decode Gazebo float32 depth images into stable binary payloads."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import math
import struct

from src.sensors.gazebo_image_codec import _message_bytes
from src.sensors.gazebo_visual_transport import GAZEBO_SIM_CLOCK, header_sequence, message_timestamp
from src.sensors.types import DepthFrame


FLOAT32_FORMATS = {13, "13", "R_FLOAT32", "FLOAT32", "r_float32", "float32"}


def parse_gazebo_depth_message(message, *, topic, staging_directory, receive_index, received_monotonic):
    width, height = int(message.get("width", 0)), int(message.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Gazebo depth dimensions must be positive")
    pixel_format = message.get("pixel_format_type", message.get("pixelFormatType"))
    if pixel_format not in FLOAT32_FORMATS:
        raise ValueError(f"unsupported Gazebo depth format: {pixel_format!r}")
    raw = _message_bytes(message.get("data"))
    step = int(message.get("step", width * 4))
    if len(raw) != step * height or step < width * 4:
        raise ValueError("Gazebo depth payload length does not match dimensions")
    packed = b"".join(raw[row * step:row * step + width * 4] for row in range(height))
    values = struct.unpack(f"<{width * height}f", packed)
    valid = sum(math.isfinite(value) and .2 <= value <= 100 for value in values)
    staging = Path(staging_directory)
    destination = staging / "depth" / f"depth_{receive_index:08d}.f32"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(packed)
    digest = sha256(packed).hexdigest()
    sequence = header_sequence(message)
    return DepthFrame(
        frame_id=f"depth-{receive_index}", source_id="gazebo_depth",
        sequence_number=receive_index if sequence is None else sequence,
        capture_timestamp=message_timestamp(message), capture_clock_domain=GAZEBO_SIM_CLOCK,
        receive_monotonic_timestamp=float(received_monotonic), width=width, height=height,
        payload_relative_path=destination.relative_to(staging).as_posix(), payload_sha256=digest,
        valid_depth_count=valid,
    )
