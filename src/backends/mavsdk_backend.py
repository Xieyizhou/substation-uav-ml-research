"""MAVSDK implementation of the research flight backend."""

from __future__ import annotations

import time

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

from src.backends.base import BackendHealth, FlightBackend, TelemetryState


async def _first(stream):
    async for value in stream:
        return value
    raise RuntimeError("telemetry stream ended without a value")


class MavsdkBackend(FlightBackend):
    backend_id = "mavsdk"

    def __init__(self, system_address="udpin://0.0.0.0:14540"):
        self.system_address = system_address
        self.drone = System()
        self._connected = False
        self._authority = False
        self._last_telemetry_monotonic = None

    async def connect(self):
        await self.drone.connect(system_address=self.system_address)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self._connected = True
                return

    async def close(self):
        if self._authority:
            await self.release_control_authority()
        self._connected = False

    async def obtain_control_authority(self):
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await self.drone.offboard.start()
        self._authority = True

    async def release_control_authority(self):
        await self.drone.offboard.stop()
        self._authority = False

    async def telemetry(self):
        position_velocity = await _first(self.drone.telemetry.position_velocity_ned())
        attitude = await _first(self.drone.telemetry.attitude_euler())
        battery = await _first(self.drone.telemetry.battery())
        armed = await _first(self.drone.telemetry.armed())
        in_air = await _first(self.drone.telemetry.in_air())
        mode = await _first(self.drone.telemetry.flight_mode())
        self._last_telemetry_monotonic = time.monotonic()
        position = position_velocity.position
        velocity = position_velocity.velocity
        return TelemetryState(
            timestamp_s=time.time(),
            north_m=position.north_m,
            east_m=position.east_m,
            down_m=position.down_m,
            velocity_north_m_s=velocity.north_m_s,
            velocity_east_m_s=velocity.east_m_s,
            velocity_down_m_s=velocity.down_m_s,
            yaw_deg=attitude.yaw_deg,
            battery_percent=battery.remaining_percent * 100.0,
            armed=bool(armed),
            in_air=bool(in_air),
            flight_mode=getattr(mode, "name", str(mode)),
        )

    async def command_velocity_ned(self, north_m_s, east_m_s, down_m_s, yaw_deg):
        if not self._authority:
            raise RuntimeError("MAVSDK Offboard control authority is not active")
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(north_m_s, east_m_s, down_m_s, yaw_deg)
        )

    async def hover(self):
        await self.command_velocity_ned(0.0, 0.0, 0.0, 0.0)

    async def land(self):
        await self.drone.action.land()

    async def return_to_home(self):
        await self.drone.action.return_to_launch()

    def health(self):
        age = (
            time.monotonic() - self._last_telemetry_monotonic
            if self._last_telemetry_monotonic is not None
            else None
        )
        return BackendHealth(
            connected=self._connected,
            control_authority=self._authority,
            telemetry_age_s=age,
            failsafe_active=False,
            message="",
        )
