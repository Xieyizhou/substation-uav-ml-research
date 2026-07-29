"""Gazebo RGB source that materializes explicit, canonical PNG CameraFrames."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from src.sensors.base import SensorSource
from src.sensors.gazebo_visual_transport import (
    GAZEBO_SIM_CLOCK,
    GAZEBO_JSON_STREAM_LIMIT_BYTES,
    RGB_CONFIGURED_TOPIC,
    RGB_MESSAGE_TYPE,
    discover_configured_topic,
    header_sequence,
    inspect_topic,
    message_timestamp,
    stop_stream_process,
    transport_environment,
)
from src.sensors.types import CameraFrame, SensorHealth


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


@dataclass(frozen=True)
class CameraSourceEvent:
    receive_index: int
    frame: CameraFrame | None = None
    invalid_reason: str | None = None

    @property
    def valid(self):
        return self.frame is not None


def _message_bytes(value):
    if not isinstance(value, str):
        raise ValueError("Gazebo image data must be base64 JSON text")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Gazebo image data is not valid base64") from error


def _pixel_description(value):
    if isinstance(value, str):
        key = value.upper()
    else:
        key = value
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
                data[offset + 2],
                data[offset + 1],
                data[offset],
                data[offset + 3],
            )
        )
    image = Image.frombytes(mode, (width, height), data)
    return image.convert("RGB")


def parse_gazebo_image_message(
    message,
    *,
    topic,
    staging_directory,
    receive_index,
    received_monotonic=None,
):
    width = int(message.get("width", 0))
    height = int(message.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Gazebo image dimensions must be positive")
    source_pixel_format, channels = _pixel_description(
        message.get("pixel_format_type", message.get("pixelFormatType"))
    )
    step = int(message.get("step", width * channels))
    source_bytes = _message_bytes(message.get("data"))
    packed = _packed_rows(source_bytes, width, height, step, channels)
    timestamp = message_timestamp(message)
    source_sequence = header_sequence(message)
    sequence = receive_index if source_sequence is None else source_sequence
    sequence_provenance = (
        "local_receive_ordinal" if source_sequence is None else "gazebo_header"
    )
    directory = Path(staging_directory)
    frames_directory = directory / "frames"
    frames_directory.mkdir(parents=True, exist_ok=True)
    relative_path = f"frames/{receive_index:09d}.png"
    destination = directory / relative_path
    image = _canonical_rgb_image(packed, width, height, source_pixel_format)
    image.save(destination, format="PNG", optimize=False, compress_level=6)
    payload = destination.read_bytes()
    return CameraFrame(
        frame_id=f"gazebo-rgb-{receive_index:09d}",
        source_id=f"gazebo_rgb:{topic}",
        sequence_number=sequence,
        capture_timestamp=timestamp,
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
            "gazebo_pixel_format": message.get(
                "pixel_format_type", message.get("pixelFormatType")
            ),
            "source_pixel_format": source_pixel_format,
            "source_row_stride_bytes": step,
            "source_payload_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "sequence_provenance": sequence_provenance,
        },
    )


class GazeboCameraSource(SensorSource):
    source_id = "gazebo_rgb"

    def __init__(self, staging_directory, *, topic="auto", stale_after_s=1.0):
        self.staging_directory = Path(staging_directory)
        self.topic = topic
        self.stale_after_s = float(stale_after_s)
        self._process = None
        self._latest = None
        self._receive_index = 0
        self._invalid_count = 0
        self._last_error = ""
        self._received_times = []

    async def start(self):
        if self._process is not None:
            return
        if self.topic == "auto":
            self.topic = await discover_configured_topic(RGB_CONFIGURED_TOPIC)
        await inspect_topic(self.topic, RGB_MESSAGE_TYPE)
        self._process = await asyncio.create_subprocess_exec(
            "gz",
            "topic",
            "-e",
            "--json-output",
            "-t",
            self.topic,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=transport_environment(),
            limit=GAZEBO_JSON_STREAM_LIMIT_BYTES,
        )

    async def events(self, *, timeout_s):
        if self._process is None:
            raise RuntimeError("Gazebo camera source has not been started")
        while True:
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=timeout_s
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError(
                    f"Gazebo RGB source timed out after {timeout_s:g}s"
                ) from error
            if not line:
                detail = (await self._process.stderr.read()).decode(
                    errors="replace"
                ).strip()
                raise RuntimeError(
                    "Gazebo RGB stream stopped" + (f": {detail}" if detail else "")
                )
            self._receive_index += 1
            received = time.monotonic()
            try:
                message = json.loads(line)
                frame = parse_gazebo_image_message(
                    message,
                    topic=self.topic,
                    staging_directory=self.staging_directory,
                    receive_index=self._receive_index,
                    received_monotonic=received,
                )
                self._latest = frame
                self._received_times.append(received)
                yield CameraSourceEvent(self._receive_index, frame=frame)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
                self._invalid_count += 1
                self._last_error = str(error)
                yield CameraSourceEvent(
                    self._receive_index,
                    invalid_reason=f"{type(error).__name__}: {error}",
                )

    async def stop(self):
        await stop_stream_process(self._process)
        self._process = None

    def latest(self):
        return self._latest

    def health(self, now_s=None):
        now_s = time.monotonic() if now_s is None else now_s
        age = (
            now_s - self._latest.receive_monotonic_timestamp
            if self._latest is not None
            else None
        )
        frequency = 0.0
        if len(self._received_times) > 1:
            duration = self._received_times[-1] - self._received_times[0]
            frequency = (
                (len(self._received_times) - 1) / duration if duration > 0 else 0.0
            )
        healthy = self._latest is not None and age <= self.stale_after_s
        message = "" if healthy else self._last_error or "waiting for RGB frame"
        return SensorHealth(
            source=self.source_id,
            healthy=healthy,
            message=message,
            frequency_hz=frequency,
            dropped_frames=self._invalid_count,
            last_frame_age_s=age,
        )
