"""Async sensor source protocol and shared readiness behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import time

from src.sensors.types import SensorHealth


class SensorSource(ABC):
    """One timestamped sensor stream with explicit lifecycle and health."""

    source_id = "unknown"

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def latest(self):
        raise NotImplementedError

    @abstractmethod
    def health(self, now_s: float | None = None) -> SensorHealth:
        raise NotImplementedError

    async def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest() is not None and self.health().healthy:
                return
            await asyncio.sleep(0.05)
        status = self.health()
        raise TimeoutError(
            f"sensor {self.source_id} did not become ready within {timeout_s:g}s: "
            f"{status.message or 'no frame'}"
        )
