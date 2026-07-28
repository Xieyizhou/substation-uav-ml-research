"""Gazebo Transport JSON subscriber for 2D laser scans."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import os
import time

from src.sensors.base import SensorSource
from src.sensors.types import LaserScanFrame, SensorHealth


SCAN_TOPIC_SUFFIXES = ("/scan", "/scan/points")


def _first(message, *keys, default=None):
    for key in keys:
        if key in message:
            return message[key]
    return default


def _timestamp_s(message: dict, fallback: float) -> float:
    header = message.get("header") or {}
    stamp = header.get("stamp") or message.get("stamp") or {}
    seconds = _first(stamp, "sec", "seconds", default=None)
    nanos = _first(stamp, "nsec", "nanoseconds", default=0)
    if seconds is None:
        return fallback
    return float(seconds) + float(nanos) / 1_000_000_000.0


def parse_laser_scan_message(
    message: dict,
    *,
    source: str,
    received_monotonic_s: float | None = None,
    sequence: int = 0,
) -> LaserScanFrame:
    """Parse Gazebo's gz.msgs.LaserScan JSON representation."""
    received = time.monotonic() if received_monotonic_s is None else received_monotonic_s
    ranges = _first(message, "ranges", "range", default=[])
    if isinstance(ranges, dict):
        ranges = ranges.get("data", [])
    ranges = tuple(float(value) for value in ranges)
    if not ranges:
        raise ValueError("laser scan contains no ranges")
    angle_min = float(_first(message, "angle_min", "angleMin", default=0.0))
    angle_max = float(_first(message, "angle_max", "angleMax", default=angle_min))
    angle_step = _first(message, "angle_step", "angleStep", default=None)
    if angle_step is None:
        angle_step = (angle_max - angle_min) / max(len(ranges) - 1, 1)
    range_min = float(_first(message, "range_min", "rangeMin", default=0.0))
    range_max = float(_first(message, "range_max", "rangeMax", default=math.inf))
    frame_id = str(_first(message, "frame", "frame_id", "frameId", default="lidar_link"))
    return LaserScanFrame(
        timestamp_s=_timestamp_s(message, fallback=time.time()),
        received_monotonic_s=received,
        frame_id=frame_id,
        angle_min_rad=angle_min,
        angle_max_rad=angle_max,
        angle_step_rad=float(angle_step),
        range_min_m=range_min,
        range_max_m=range_max,
        ranges_m=ranges,
        source=source,
        sequence=sequence,
    )


def _transport_environment():
    environment = os.environ.copy()
    environment.setdefault("GZ_IP", "127.0.0.1")
    return environment


async def discover_lidar_topic(timeout_s: float = 5.0) -> str:
    process = await asyncio.create_subprocess_exec(
        "gz",
        "topic",
        "-l",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_transport_environment(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError("timed out while discovering Gazebo topics")
    if process.returncode:
        raise RuntimeError(
            "Gazebo topic discovery failed: " + stderr.decode(errors="replace").strip()
        )
    topics = [line.strip() for line in stdout.decode().splitlines() if line.strip()]
    candidates = [
        topic
        for topic in topics
        if any(topic.endswith(suffix) for suffix in SCAN_TOPIC_SUFFIXES)
        or "lidar" in topic.lower()
    ]
    scan_candidates = [topic for topic in candidates if topic.endswith("/scan")]
    candidates = scan_candidates or candidates
    if not candidates:
        raise RuntimeError("no Gazebo LiDAR scan topic was found")
    return sorted(
        candidates,
        key=lambda topic: (
            "lidar_2d" not in topic.lower(),
            "gpu_lidar" not in topic.lower(),
            topic,
        ),
    )[0]


class GazeboLidar2DSource(SensorSource):
    source_id = "gazebo_lidar_2d"

    def __init__(self, topic="auto", stale_after_s=0.5):
        self.topic = topic
        self.stale_after_s = float(stale_after_s)
        self._process = None
        self._reader_task = None
        self._latest = None
        self._error = ""
        self._sequence = 0
        self._received_times = deque(maxlen=60)
        self._dropped_frames = 0

    async def start(self):
        if self._reader_task is not None:
            return
        if self.topic == "auto":
            self.topic = await discover_lidar_topic()
        self._process = await asyncio.create_subprocess_exec(
            "gz",
            "topic",
            "-e",
            "--json-output",
            "-t",
            self.topic,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_transport_environment(),
        )
        self._reader_task = asyncio.create_task(self._read_messages(), name="gazebo-lidar")

    async def _read_messages(self):
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    stderr = await self._process.stderr.read()
                    raise RuntimeError(
                        "Gazebo LiDAR stream stopped"
                        + (f": {stderr.decode(errors='replace').strip()}" if stderr else "")
                    )
                try:
                    message = json.loads(line)
                    self._sequence += 1
                    now = time.monotonic()
                    self._latest = parse_laser_scan_message(
                        message,
                        source=self.source_id,
                        received_monotonic_s=now,
                        sequence=self._sequence,
                    )
                    self._received_times.append(now)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self._dropped_frames += 1
                    self._error = f"invalid scan message: {error}"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._error = str(error)

    async def stop(self):
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    def latest(self):
        return self._latest

    def health(self, now_s=None):
        now_s = time.monotonic() if now_s is None else now_s
        age = self._latest.age_s(now_s) if self._latest else None
        frequency = 0.0
        if len(self._received_times) >= 2:
            duration = self._received_times[-1] - self._received_times[0]
            frequency = (len(self._received_times) - 1) / duration if duration > 0 else 0.0
        healthy = bool(
            self._latest is not None
            and age is not None
            and age <= self.stale_after_s
            and not self._error
        )
        message = self._error
        if not message and self._latest is None:
            message = "waiting for first scan"
        elif not message and age is not None and age > self.stale_after_s:
            message = f"scan is stale ({age:.3f}s)"
        return SensorHealth(
            source=self.source_id,
            healthy=healthy,
            message=message,
            frequency_hz=frequency,
            dropped_frames=self._dropped_frames,
            last_frame_age_s=age,
        )
