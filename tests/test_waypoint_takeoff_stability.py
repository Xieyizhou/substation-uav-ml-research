from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src.flight import waypoint_executor
from src.flight.takeoff_stability import (
    wait_for_ground_stability,
    wait_for_takeoff_hover,
)
from src.flight.waypoint_executor import (
    takeoff_climb_waypoint,
    validate_takeoff_stability,
)


def _latest(
    *, north=0.0, east=0.0, down=-1.5, roll=0.0, pitch=0.0,
    velocity_north=0.0, velocity_east=0.0, velocity_down=0.0,
):
    return {
        "position_velocity": SimpleNamespace(
            position=SimpleNamespace(
                north_m=north,
                east_m=east,
                down_m=down,
            ),
            velocity=SimpleNamespace(
                north_m_s=velocity_north,
                east_m_s=velocity_east,
                down_m_s=velocity_down,
            ),
        ),
        "attitude": SimpleNamespace(roll_deg=roll, pitch_deg=pitch),
    }


class TakeoffStabilityTests(unittest.TestCase):
    def test_accepts_stable_takeoff(self):
        validate_takeoff_stability(_latest(), 1.5)
        validate_takeoff_stability(_latest(down=-0.1), 1.5)

    def test_rejects_excessive_attitude(self):
        with self.assertRaisesRegex(RuntimeError, "Takeoff is unstable"):
            validate_takeoff_stability(_latest(roll=45.0), 1.5)

    def test_rejects_horizontal_drift(self):
        with self.assertRaisesRegex(RuntimeError, "Takeoff drifted"):
            validate_takeoff_stability(_latest(north=3.0), 1.5)

    def test_rejects_implausible_altitude(self):
        with self.assertRaisesRegex(RuntimeError, "stability envelope"):
            validate_takeoff_stability(_latest(down=-4.0), 1.5)

    def test_takeoff_climb_holds_horizontal_position(self):
        waypoint = takeoff_climb_waypoint(
            _latest(north=0.2, east=-0.1),
            -1.5,
        )
        self.assertEqual(
            waypoint,
            {
                "name": "TAKEOFF_CLIMB",
                "north_m": 0.2,
                "east_m": -0.1,
                "down_m": -1.5,
            },
        )


class ContinuousStabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_ground_gate_requires_level_stationary_vehicle(self):
        latest = _latest(down=0.0)
        with patch(
            "src.flight.takeoff_stability._wait_for_stable_window",
            AsyncMock(),
        ) as wait:
            await wait_for_ground_stability(latest, 10.0)
        self.assertEqual(wait.await_args.kwargs["stage"], "ground")
        self.assertTrue(wait.await_args.args[1]())
        latest["position_velocity"].velocity.east_m_s = 0.2
        self.assertFalse(wait.await_args.args[1]())

    async def test_hover_gate_requires_altitude_and_low_velocity(self):
        latest = _latest(down=-2.0)
        with patch(
            "src.flight.takeoff_stability._wait_for_stable_window",
            AsyncMock(),
        ) as wait:
            await wait_for_takeoff_hover(latest, 2.5, 10.0)
        self.assertEqual(wait.await_args.kwargs["timeout_s"], 30.0)
        self.assertEqual(wait.await_args.kwargs["stage"], "takeoff hover")
        self.assertTrue(wait.await_args.args[1]())
        latest["position_velocity"].position.down_m = -0.5
        self.assertFalse(wait.await_args.args[1]())

    async def test_mission_waits_for_stable_hover_before_offboard(self):
        action = SimpleNamespace(
            set_takeoff_altitude=AsyncMock(), takeoff=AsyncMock(), land=AsyncMock()
        )
        offboard = SimpleNamespace(
            set_velocity_ned=AsyncMock(), start=AsyncMock(), stop=AsyncMock()
        )
        drone = SimpleNamespace(action=action, offboard=offboard)
        waypoint = {
            "name": "WP01", "north_m": 0.5,
            "east_m": 0.5, "down_m": -1.5,
        }
        with (
            patch.object(waypoint_executor, "arm_when_ready", new=AsyncMock()),
            patch.object(waypoint_executor, "wait_for_local_position", new=AsyncMock()),
            patch.object(waypoint_executor, "wait_for_takeoff_hover", new=AsyncMock()) as hover,
            patch.object(waypoint_executor, "fly_to_waypoint", new=AsyncMock()),
            patch.object(
                waypoint_executor, "fly_waypoint_route",
                new=AsyncMock(return_value=[waypoint]),
            ),
            patch.object(waypoint_executor, "hover_at_waypoint", new=AsyncMock()),
            patch.object(waypoint_executor, "wait_until_landed", new=AsyncMock()),
        ):
            await waypoint_executor.fly_astar_waypoints(
                drone, _latest(down=-1.0), {}, {}, [waypoint], {}, None, {}, {},
            )
        hover.assert_awaited_once_with(
            unittest.mock.ANY, 1.5, waypoint_executor.TELEMETRY_TIMEOUT_S
        )
        offboard.start.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
