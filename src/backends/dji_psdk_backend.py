"""Client boundary for the external DJI PSDK C++ bridge."""

from __future__ import annotations

from src.backends.base import BackendHealth, FlightBackend, TelemetryState


class DjiPsdkBackend(FlightBackend):
    """Typed wrapper around generated gRPC stubs.

    The DJI SDK is intentionally isolated in the companion-computer bridge.
    This module remains importable without DJI or gRPC dependencies so PX4
    simulation and offline ML work stay portable.
    """

    backend_id = "dji_psdk"

    def __init__(self, target="127.0.0.1:50051"):
        self.target = target
        self._channel = None
        self._stub = None
        self._last_health = BackendHealth(False, False, None, False, "not connected")

    async def connect(self):
        try:
            import grpc
            from integrations.dji_psdk.generated import (
                flight_bridge_pb2,
                flight_bridge_pb2_grpc,
            )
        except ImportError as error:
            raise RuntimeError(
                "DJI backend requires grpcio and generated PSDK bridge stubs"
            ) from error
        self._pb = flight_bridge_pb2
        self._channel = grpc.aio.insecure_channel(self.target)
        self._stub = flight_bridge_pb2_grpc.FlightBridgeStub(self._channel)
        response = await self._stub.Health(self._pb.Empty())
        self._last_health = BackendHealth(
            bool(response.connected),
            bool(response.control_authority),
            response.telemetry_age_s,
            bool(response.failsafe_active),
            response.message,
        )

    async def close(self):
        if self._channel is not None:
            await self._channel.close()
        self._channel = None
        self._stub = None

    def _require_stub(self):
        if self._stub is None:
            raise RuntimeError("DJI PSDK bridge is not connected")

    async def obtain_control_authority(self):
        self._require_stub()
        await self._stub.ObtainControl(self._pb.Empty())

    async def release_control_authority(self):
        self._require_stub()
        await self._stub.ReleaseControl(self._pb.Empty())

    async def telemetry(self):
        self._require_stub()
        state = await self._stub.LatestTelemetry(self._pb.Empty())
        return TelemetryState(
            timestamp_s=state.timestamp_s,
            north_m=state.north_m,
            east_m=state.east_m,
            down_m=state.down_m,
            velocity_north_m_s=state.velocity_north_m_s,
            velocity_east_m_s=state.velocity_east_m_s,
            velocity_down_m_s=state.velocity_down_m_s,
            yaw_deg=state.yaw_deg,
            battery_percent=state.battery_percent,
            armed=state.armed,
            in_air=state.in_air,
            flight_mode=state.flight_mode,
        )

    async def command_velocity_ned(self, north_m_s, east_m_s, down_m_s, yaw_deg):
        self._require_stub()
        await self._stub.CommandVelocity(
            self._pb.VelocityNedYaw(
                north_m_s=north_m_s,
                east_m_s=east_m_s,
                down_m_s=down_m_s,
                yaw_deg=yaw_deg,
            )
        )

    async def hover(self):
        self._require_stub()
        await self._stub.Hover(self._pb.Empty())

    async def land(self):
        self._require_stub()
        await self._stub.Land(self._pb.Empty())

    async def return_to_home(self):
        self._require_stub()
        await self._stub.ReturnToHome(self._pb.Empty())

    def health(self):
        return self._last_health
