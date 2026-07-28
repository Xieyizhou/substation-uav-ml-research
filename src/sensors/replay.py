"""Deterministic JSONL laser-scan recording and replay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

from src.sensors.base import SensorSource
from src.sensors.types import LaserScanFrame, SensorHealth


def load_scan_records(path: Path) -> list[LaserScanFrame]:
    frames = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "laser_scan_2d":
                raise ValueError(f"{path}:{line_number}: unsupported record type")
            frames.append(LaserScanFrame.from_record(record))
    if not frames:
        raise ValueError(f"{path} contains no laser scan records")
    return frames


def append_scan_record(path: Path, frame: LaserScanFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(frame.to_record(), separators=(",", ":")) + "\n")


class ReplayLidarSource(SensorSource):
    source_id = "replay"

    def __init__(self, path: Path, rate=1.0, loop=False, stale_after_s=0.5):
        self.path = Path(path)
        self.rate = float(rate)
        self.loop = bool(loop)
        self.stale_after_s = float(stale_after_s)
        self._frames = []
        self._latest = None
        self._task = None
        self._error = ""
        self._count = 0

    async def start(self):
        if self.rate <= 0:
            raise ValueError("replay rate must be positive")
        self._frames = load_scan_records(self.path)
        self._task = asyncio.create_task(self._run(), name="lidar-replay")

    async def _run(self):
        try:
            while True:
                previous_timestamp = None
                for frame in self._frames:
                    if previous_timestamp is not None:
                        delay = max(0.0, frame.timestamp_s - previous_timestamp) / self.rate
                        await asyncio.sleep(delay)
                    self._count += 1
                    self._latest = LaserScanFrame(
                        **{
                            **frame.__dict__,
                            "received_monotonic_s": time.monotonic(),
                            "source": self.source_id,
                            "sequence": self._count,
                        }
                    )
                    previous_timestamp = frame.timestamp_s
                if not self.loop:
                    return
                # A one-frame fixture has no inter-frame delay. Always yield
                # between loops so replay cannot starve the flight event loop.
                await asyncio.sleep(max(0.01, 0.05 / self.rate))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._error = str(error)

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def latest(self):
        return self._latest

    def health(self, now_s=None):
        now_s = time.monotonic() if now_s is None else now_s
        age = self._latest.age_s(now_s) if self._latest else None
        healthy = self._latest is not None and not self._error
        return SensorHealth(
            source=self.source_id,
            healthy=healthy,
            message=self._error or ("" if healthy else "waiting for replay frame"),
            frequency_hz=0.0,
            dropped_frames=0,
            last_frame_age_s=age,
        )
