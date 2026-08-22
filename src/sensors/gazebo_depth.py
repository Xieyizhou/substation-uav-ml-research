"""Gazebo depth-camera source with health and stale-frame accounting."""

import asyncio
import json
import time

from src.sensors.base import SensorSource
from src.sensors.gazebo_depth_codec import parse_gazebo_depth_message
from src.sensors.gazebo_visual_transport import (
    DEPTH_CONFIGURED_TOPIC, DEPTH_MESSAGE_TYPE, GAZEBO_JSON_STREAM_LIMIT_BYTES,
    discover_configured_topic, inspect_topic, stop_stream_process, transport_environment,
)
from src.sensors.types import SensorHealth


class GazeboDepthSource(SensorSource):
    source_id = "gazebo_depth"

    def __init__(self, staging_directory, *, topic="auto", stale_after_s=1.0):
        from pathlib import Path
        self.staging_directory, self.topic = Path(staging_directory), topic
        self.stale_after_s = float(stale_after_s)
        self._process = self._latest = None
        self._receive_index = self._invalid_count = 0
        self._last_error = ""
        self._received_times = []

    async def start(self):
        if self._process is not None:
            return
        if self.topic == "auto":
            self.topic = await discover_configured_topic(DEPTH_CONFIGURED_TOPIC)
        await inspect_topic(self.topic, DEPTH_MESSAGE_TYPE)
        self._process = await asyncio.create_subprocess_exec(
            "gz", "topic", "-e", "--json-output", "-t", self.topic,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=transport_environment(), limit=GAZEBO_JSON_STREAM_LIMIT_BYTES,
        )

    async def events(self, *, timeout_s):
        if self._process is None:
            raise RuntimeError("Gazebo depth source has not been started")
        while True:
            try:
                line = await asyncio.wait_for(self._process.stdout.readline(), timeout_s)
            except asyncio.TimeoutError as error:
                raise TimeoutError(f"Gazebo depth source timed out after {timeout_s:g}s") from error
            if not line:
                raise RuntimeError("Gazebo depth stream stopped")
            self._receive_index += 1
            received = time.monotonic()
            try:
                frame = parse_gazebo_depth_message(
                    json.loads(line), topic=self.topic, staging_directory=self.staging_directory,
                    receive_index=self._receive_index, received_monotonic=received,
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
                self._invalid_count += 1
                self._last_error = f"{type(error).__name__}: {error}"
                continue
            self._latest = frame
            self._received_times.append(received)
            yield frame

    async def stop(self):
        await stop_stream_process(self._process)
        self._process = None

    def latest(self):
        return self._latest

    def health(self, now_s=None):
        now_s = time.monotonic() if now_s is None else now_s
        age = now_s - self._latest.receive_monotonic_timestamp if self._latest else None
        duration = self._received_times[-1] - self._received_times[0] if len(self._received_times) > 1 else 0
        frequency = (len(self._received_times) - 1) / duration if duration > 0 else 0
        healthy = self._latest is not None and age <= self.stale_after_s
        return SensorHealth(self.source_id, healthy, "" if healthy else self._last_error or "waiting for depth frame", frequency, self._invalid_count, age)
