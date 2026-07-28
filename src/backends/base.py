"""High-level flight backend contract used by the research runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryState:
    timestamp_s: float
    north_m: float
    east_m: float
    down_m: float
    velocity_north_m_s: float
    velocity_east_m_s: float
    velocity_down_m_s: float
    yaw_deg: float
    battery_percent: float
    armed: bool
    in_air: bool
    flight_mode: str


@dataclass(frozen=True)
class BackendHealth:
    connected: bool
    control_authority: bool
    telemetry_age_s: float | None
    failsafe_active: bool
    message: str = ""


class FlightBackend(ABC):
    """Commands are high level; ML code never receives motor-level control."""

    backend_id = "unknown"

    @abstractmethod
    async def connect(self):
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        raise NotImplementedError

    @abstractmethod
    async def obtain_control_authority(self):
        raise NotImplementedError

    @abstractmethod
    async def release_control_authority(self):
        raise NotImplementedError

    @abstractmethod
    async def telemetry(self) -> TelemetryState:
        raise NotImplementedError

    @abstractmethod
    async def command_velocity_ned(self, north_m_s, east_m_s, down_m_s, yaw_deg):
        raise NotImplementedError

    @abstractmethod
    async def hover(self):
        raise NotImplementedError

    @abstractmethod
    async def land(self):
        raise NotImplementedError

    @abstractmethod
    async def return_to_home(self):
        raise NotImplementedError

    @abstractmethod
    def health(self) -> BackendHealth:
        raise NotImplementedError
