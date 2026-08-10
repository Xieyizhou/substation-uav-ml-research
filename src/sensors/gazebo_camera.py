"""Gazebo RGB source that materializes explicit, canonical PNG CameraFrames."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time

from src.sensors.base import SensorSource
from src.sensors.gazebo_image_codec import parse_gazebo_image_message
from src.sensors.gazebo_visual_transport import (
    GAZEBO_JSON_STREAM_LIMIT_BYTES,
    RGB_CONFIGURED_TOPIC,
    RGB_MESSAGE_TYPE,
    discover_configured_topic,
    inspect_topic,
    stop_stream_process,
    transport_environment,
)
from src.sensors.types import CameraFrame, SensorHealth


@dataclass(frozen=True)
class CameraSourceEvent:
    receive_index: int
    frame: CameraFrame | None = None
    invalid_reason: str | None = None

    @property
    def valid(self):
        return self.frame is not None


class GazeboCameraSource(SensorSource):
    source_id = "gazebo_rgb"

    def __init__(
        self, staging_directory, *, topic="auto", stale_after_s=1.0,
        encoder_workers=4,
    ):
        if not 1 <= int(encoder_workers) <= 16:
            raise ValueError("encoder_workers must be between 1 and 16")
        self.staging_directory = Path(staging_directory)
        self.topic = topic
        self.stale_after_s = float(stale_after_s)
        self._process = None
        self._latest = None
        self._receive_index = 0
        self._invalid_count = 0
        self._last_error = ""
        self._received_times = []
        self.encoder_workers = int(encoder_workers)

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

    async def _read_event(self, timeout_s):
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
        receive_index = self._receive_index
        received = time.monotonic()

        def decode():
            try:
                message = json.loads(line)
                frame = parse_gazebo_image_message(
                    message,
                    topic=self.topic,
                    staging_directory=self.staging_directory,
                    receive_index=receive_index,
                    received_monotonic=received,
                )
                return CameraSourceEvent(receive_index, frame=frame)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
                return CameraSourceEvent(
                    receive_index,
                    invalid_reason=f"{type(error).__name__}: {error}",
                )

        return asyncio.create_task(asyncio.to_thread(decode))

    async def events(self, *, timeout_s):
        if self._process is None:
            raise RuntimeError("Gazebo camera source has not been started")
        pending = deque()
        try:
            while True:
                if not pending:
                    for _ in range(self.encoder_workers):
                        pending.append(await self._read_event(timeout_s))
                event = await pending.popleft()
                if event.valid:
                    frame = event.frame
                    self._latest = frame
                    self._received_times.append(frame.receive_monotonic_timestamp)
                else:
                    self._invalid_count += 1
                    self._last_error = event.invalid_reason or "invalid frame"
                yield event
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

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
