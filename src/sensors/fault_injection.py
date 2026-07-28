"""Deterministic LiDAR noise, point dropout, and stream outage injection."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random

from src.sensors.base import SensorSource
from src.sensors.types import LaserScanFrame, SensorHealth


class FaultInjectedLidarSource(SensorSource):
    source_id = "fault_injected_lidar"

    def __init__(self, source, manifest):
        self.source = source
        self.manifest = manifest
        self.seed = int(manifest["seed"])
        self._cached_sequence = None
        self._cached_frame = None
        self._outage = False
        self._fault_drops = 0

    @classmethod
    def from_path(cls, source, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls(source, json.load(handle))

    async def start(self):
        await self.source.start()

    async def stop(self):
        await self.source.stop()

    async def wait_ready(self, timeout_s):
        return await self.source.wait_ready(timeout_s)

    def _inject(self, frame):
        generator = random.Random(self.seed + frame.sequence * 104729)
        outage_probability = float(
            self.manifest.get("sensor_outage_probability", 0.0)
        )
        self._outage = generator.random() < outage_probability
        if self._outage:
            self._fault_drops += 1
            return None
        noise = float(self.manifest.get("lidar_noise_stddev_m", 0.0))
        dropout = float(self.manifest.get("lidar_dropout_probability", 0.0))
        values = []
        for value in frame.ranges_m:
            if generator.random() < dropout:
                values.append(float("inf"))
                self._fault_drops += 1
            elif math.isfinite(value):
                values.append(
                    min(
                        max(value + generator.gauss(0.0, noise), frame.range_min_m),
                        frame.range_max_m,
                    )
                )
            else:
                values.append(value)
        return LaserScanFrame(
            **{
                **frame.__dict__,
                "ranges_m": tuple(values),
                "source": self.source_id,
            }
        )

    def latest(self):
        frame = self.source.latest()
        if frame is None:
            return None
        if frame.sequence != self._cached_sequence:
            self._cached_sequence = frame.sequence
            self._cached_frame = self._inject(frame)
        return self._cached_frame

    def health(self, now_s=None):
        underlying = self.source.health(now_s)
        self.latest()
        if self._outage:
            return SensorHealth(
                source=self.source_id,
                healthy=False,
                message="injected sensor stream outage",
                frequency_hz=underlying.frequency_hz,
                dropped_frames=underlying.dropped_frames + self._fault_drops,
                last_frame_age_s=underlying.last_frame_age_s,
            )
        return SensorHealth(
            source=self.source_id,
            healthy=underlying.healthy,
            message=underlying.message,
            frequency_hz=underlying.frequency_hz,
            dropped_frames=underlying.dropped_frames + self._fault_drops,
            last_frame_age_s=underlying.last_frame_age_s,
        )
