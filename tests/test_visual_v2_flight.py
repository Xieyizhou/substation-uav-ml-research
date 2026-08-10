from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.flight.waypoint_executor import hover_at_waypoint
from src.vision.collection.flight_route import (
    _fly_observations,
    _return_and_land,
    _return_waypoints,
    _settle_departure_yaw,
    _settle_observation_yaw,
    _takeoff,
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

    async def test_yaw_slew_progresses_when_attitude_response_lags(self):
        route = replace(
            self.route(),
            yaw_acquisition_timeout_s=1.0,
            yaw_settle_duration_s=0.001,
        )
        observation = replace(route.waypoints[0], yaw_deg=40.0)
        drone = Mock()
        latest = {
            "attitude": SimpleNamespace(
                yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0,
            )
        }
        command_count = 0

        async def accept_after_lag(_command):
            nonlocal command_count
            command_count += 1
            if command_count >= 3:
                latest["attitude"].yaw_deg = 40.0

        drone.offboard.set_velocity_ned = AsyncMock(side_effect=accept_after_lag)
        with patch(
            "src.vision.collection.flight_route.YAW_COMMAND_INTERVAL_S",
            0.001,
        ):
            await _settle_observation_yaw(
                drone, latest, {}, observation, route,
            )
        commands = [
            call.args[0].yaw_deg
            for call in drone.offboard.set_velocity_ned.await_args_list
        ]
        self.assertGreaterEqual(len(commands), 2)
        self.assertGreater(commands[1], commands[0])

    async def test_departure_heading_settles_before_route_phase_begins(self):
        route = self.route()
        phase_state = {}
        with patch(
            "src.vision.collection.flight_route._settle_observation_yaw",
            AsyncMock(),
        ) as settle:
            await _settle_departure_yaw(Mock(), {}, phase_state, route)
        departure = settle.await_args.args[3]
        self.assertEqual(departure.yaw_deg, route.waypoints[0].yaw_deg)
        self.assertEqual(
            departure.waypoint_id,
            f"departure_{route.waypoints[0].waypoint_id}",
        )
        self.assertEqual(phase_state["phase"], "cruise_distant")

    def test_yaw_error_wraps_at_signed_boundary(self):
        self.assertEqual(_yaw_error_deg(-179.0, 180.0), -1.0)
        self.assertEqual(_yaw_error_deg(179.0, -179.0), 2.0)

    async def test_visual_takeoff_uses_action_before_offboard_descent(self):
        route = self.route()
        drone = Mock()
        drone.action.arm = AsyncMock()
        drone.action.takeoff = AsyncMock()
        drone.action.set_takeoff_altitude = AsyncMock()
        drone.param.set_param_float = AsyncMock()
        drone.offboard.set_velocity_ned = AsyncMock()
        drone.offboard.start = AsyncMock()
        latest = {
            "position_velocity": SimpleNamespace(
                position=SimpleNamespace(north_m=0.1, east_m=-0.2, down_m=0.0)
            ),
            "attitude": SimpleNamespace(
                yaw_deg=96.0, roll_deg=0.0, pitch_deg=0.0
            ),
        }
        configs = ({}, None, {"mode": "disabled"}, {})
        with patch(
            "src.vision.collection.flight_route.fly_to_waypoint",
            AsyncMock(),
        ) as fly_to, patch(
            "src.vision.collection.flight_route.wait_for_ground_stability",
            AsyncMock(),
        ) as ground_stable, patch(
            "src.vision.collection.flight_route.wait_for_takeoff_hover",
            AsyncMock(),
        ) as hover_stable:
            await _takeoff(drone, latest, {}, {}, route, configs)
        ground_stable.assert_awaited_once()
        hover_stable.assert_awaited_once()
        drone.param.set_param_float.assert_awaited_once_with("MPC_TKO_SPEED", 0.5)
        drone.action.set_takeoff_altitude.assert_awaited_once_with(2.5)
        drone.action.arm.assert_awaited_once()
        drone.action.takeoff.assert_awaited_once()
        drone.offboard.start.assert_awaited_once()
        initial_setpoint = drone.offboard.set_velocity_ned.await_args.args[0]
        self.assertEqual(initial_setpoint.yaw_deg, 96.0)
        takeoff_waypoint = fly_to.await_args.args[4]
        self.assertEqual(takeoff_waypoint["north_m"], 0.1)
        self.assertEqual(takeoff_waypoint["east_m"], -0.2)
        self.assertEqual(takeoff_waypoint["down_m"], -1.5)
        self.assertEqual(takeoff_waypoint["yaw_deg"], 96.0)

    def test_return_waypoints_only_use_frozen_astar_path(self):
        route = self.route()
        waypoints = _return_waypoints(route)
        self.assertEqual(len(waypoints), len(route.return_transit_cells))
        self.assertEqual(waypoints[-1]["name"], "return_start")
        self.assertEqual(
            (waypoints[-1]["east_m"], waypoints[-1]["north_m"]),
            (0.0, 0.0),
        )
        self.assertEqual(
            [
                (
                    waypoint["east_m"] + route.start_cell[0],
                    waypoint["north_m"] + route.start_cell[1],
                )
                for waypoint in waypoints
            ],
            [tuple(map(float, cell)) for cell in route.return_transit_cells],
        )

    async def test_observation_route_is_relative_to_spawn_cell(self):
        route = self.route()
        drone = Mock()
        configs = ({}, None, {"mode": "disabled"}, {})
        with (
            patch(
                "src.vision.collection.flight_route.fly_waypoint_route",
                AsyncMock(),
            ) as fly_route,
            patch(
                "src.vision.collection.flight_route._settle_observation_yaw",
                AsyncMock(),
            ),
            patch(
                "src.vision.collection.flight_route.hover_at_waypoint",
                AsyncMock(),
            ),
        ):
            await _fly_observations(drone, {}, {}, {}, route, configs)
        first_observation = route.waypoints[0]
        first_executed_route = fly_route.await_args_list[0].args[4]
        final_waypoint = first_executed_route[-1]
        self.assertAlmostEqual(
            final_waypoint["east_m"],
            first_observation.east_m - route.start_cell[0] - 0.5,
        )
        self.assertAlmostEqual(
            final_waypoint["north_m"],
            first_observation.north_m - route.start_cell[1] - 0.5,
        )

    async def test_return_execution_consumes_frozen_path_then_lands(self):
        route = self.route()
        drone = Mock()
        drone.offboard.stop = AsyncMock()
        drone.action.land = AsyncMock()
        configs = ({}, None, {"mode": "disabled"}, {})
        with (
            patch(
                "src.vision.collection.flight_route.fly_waypoint_route",
                AsyncMock(),
            ) as fly_route,
            patch(
                "src.vision.collection.flight_route.wait_until_landed",
                AsyncMock(),
            ),
        ):
            await _return_and_land(
                drone,
                {},
                {},
                {},
                route,
                configs,
            )
        executed = fly_route.await_args.args[4]
        self.assertEqual(executed, _return_waypoints(route))
        self.assertEqual(fly_route.await_args.args[6], "return")
        drone.offboard.stop.assert_awaited_once()
        drone.action.land.assert_awaited_once()

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
