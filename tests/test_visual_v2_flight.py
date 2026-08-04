from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.flight.waypoint_executor import hover_at_waypoint
from src.vision.collection.flight_route import (
    _settle_departure_yaw,
    _settle_observation_yaw,
    _yaw_error_deg,
)
from src.vision.collection.layout import build_layout_manifest
from src.vision.collection.route import build_visual_route
from src.vision.contracts.protocol import load_protocol


class VisualYawSettlingTests(unittest.IsolatedAsyncioTestCase):
    def route(self):
        protocol = load_protocol("v2")
        layout = build_layout_manifest(
            "development-01",
            "development",
            3001,
            protocol["randomization"]["configuration"],
        )
        route = build_visual_route(
            layout,
            "switchgear_centered_v2",
            "switchgear",
        )
        return replace(
            route,
            yaw_settle_duration_s=0.01,
            yaw_acquisition_timeout_s=1.0,
        )

    async def test_hold_boundary_is_published_only_after_yaw_is_stable(self):
        route = self.route()
        observation = replace(route.waypoints[0], yaw_deg=180.0)
        drone = Mock()
        drone.offboard.set_velocity_ned = AsyncMock()
        publisher = Mock()
        await _settle_observation_yaw(
            drone,
            {"attitude": SimpleNamespace(
                yaw_deg=-179.0, roll_deg=0.5, pitch_deg=-0.5,
            )},
            {"_event_publisher": publisher},
            observation,
            route,
        )
        self.assertGreaterEqual(drone.offboard.set_velocity_ned.await_count, 2)
        publisher.publish.assert_called_once()
        self.assertEqual(publisher.publish.call_args.args[0], "yaw_settled")

    async def test_departure_heading_settles_before_route_phase_begins(self):
        route = self.route()
        with patch(
            "src.vision.collection.flight_route._settle_observation_yaw",
            AsyncMock(),
        ) as settle:
            await _settle_departure_yaw(Mock(), {}, {}, route)
        departure = settle.await_args.args[3]
        self.assertEqual(departure.yaw_deg, route.waypoints[0].yaw_deg)
        self.assertEqual(
            departure.waypoint_id,
            f"departure_{route.waypoints[0].waypoint_id}",
        )

    def test_yaw_error_wraps_at_signed_boundary(self):
        self.assertEqual(_yaw_error_deg(-179.0, 180.0), -1.0)
        self.assertEqual(_yaw_error_deg(179.0, -179.0), 2.0)

    async def test_hover_normalizes_unsigned_yaw(self):
        drone = Mock()
        drone.offboard.set_velocity_ned = AsyncMock()
        phase_state = {}
        target_state = {}
        with patch("src.flight.waypoint_executor.asyncio.sleep", AsyncMock()):
            await hover_at_waypoint(
                drone,
                phase_state,
                target_state,
                {"name": "hold", "yaw_deg": 270.0},
                "close_inspection",
                1.0,
            )
        command = drone.offboard.set_velocity_ned.await_args.args[0]
        self.assertEqual(command.yaw_deg, -90.0)


if __name__ == "__main__":
    unittest.main()
