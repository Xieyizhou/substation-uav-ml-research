import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.flight import waypoint_executor
from src.flight.replanning_controller import (
    attempt_local_replan,
    empty_replan_state,
    route_allows_local_replan,
    should_attempt_local_replan,
    update_replanned_route_escape,
)


class ReplanningControllerTests(unittest.TestCase):
    def test_return_route_still_stops_for_geometric_obstacle_evidence(self):
        detection = {
            "nearest_obstacle": {"distance_m": 1.0},
            "detected_obstacles": [],
            "dynamic_grid_cells": [],
        }
        self.assertTrue(
            waypoint_executor.return_has_geometric_obstacle_evidence(
                "return", detection
            )
        )
        self.assertFalse(
            waypoint_executor.return_has_geometric_obstacle_evidence(
                "outbound", detection
            )
        )

    def test_active_escape_suppresses_duplicate_replans_until_risk_clears(self):
        config = {"risk_level": "danger", "enabled": True, "max_replans": 3,
                  "cooldown_s": 0}
        state = {"following_replanned_route": True, "replan_count": 1,
                 "last_attempt_time": None}
        self.assertTrue(update_replanned_route_escape(config, state, "danger"))
        self.assertFalse(
            should_attempt_local_replan(config, state, "danger", 1.0)
        )
        self.assertFalse(update_replanned_route_escape(config, state, "warning"))
        self.assertTrue(
            should_attempt_local_replan(config, state, "danger", 2.0)
        )

    def test_active_replan_only_changes_outbound_route(self):
        config = {"mode": "active"}
        self.assertTrue(route_allows_local_replan(config, "outbound"))
        self.assertFalse(route_allows_local_replan(config, "return"))
        self.assertTrue(route_allows_local_replan({"mode": "log_only"}, "return"))

    @patch("src.flight.replanning_controller.astar", return_value=[(1, 1), (7, 7)])
    def test_known_static_observation_is_not_inflated_twice(self, astar):
        config = {
            "goal_cell": (7, 7),
            "static_obstacles": {(3, 3)},
            "dynamic_obstacle_inflation": 1,
            "width": 10,
            "height": 10,
            "allow_diagonal": True,
            "resolution_m": 1.0,
        }
        detection = {"dynamic_grid_cells": [(3, 3), (6, 6)]}
        path = attempt_local_replan(
            config,
            empty_replan_state(),
            SimpleNamespace(east_m=1.0, north_m=1.0),
            detection,
            1.0,
        )
        self.assertEqual(path, [(1, 1), (7, 7)])
        obstacles = astar.call_args.kwargs["obstacles"]
        self.assertIn((3, 3), obstacles)
        self.assertNotIn((2, 2), obstacles)
        self.assertIn((5, 5), obstacles)


class ReplannedRouteHistoryTests(unittest.IsolatedAsyncioTestCase):
    @patch("src.flight.waypoint_executor.publish_mission_event")
    @patch("src.flight.waypoint_executor.fly_to_waypoint")
    async def test_completed_route_preserves_prefix_and_replacement(self, fly, _publish):
        first = {"name": "WP01"}
        second = {"name": "WP02"}
        replacement = [{"name": "RWP01"}, {"name": "RWP02"}]
        fly.side_effect = [None, replacement, None, None]
        completed = await waypoint_executor.fly_waypoint_route(
            object(), {}, {}, {}, [first, second], "outbound_to_goal", "outbound",
            {}, None, {"mode": "active"}, {},
        )
        self.assertEqual(
            [waypoint["name"] for waypoint in completed],
            ["WP01", "RWP01", "RWP02"],
        )

    @patch("src.flight.waypoint_executor.asyncio.sleep", return_value=None)
    @patch("src.flight.waypoint_executor.current_perception_detection")
    async def test_return_route_does_not_hover_forever_on_model_only_danger(
        self, detection, _sleep
    ):
        detection.return_value = {
            "risk_level": "danger", "sensor_healthy": True,
            "sensor_frame_age_s": 0.0, "dynamic_grid_cells": [],
            "detected_obstacles": [], "nearest_obstacle": None,
        }
        drone = SimpleNamespace(
            offboard=SimpleNamespace(set_velocity_ned=AsyncMock())
        )
        position = SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.5)
        latest = {
            "position_velocity": SimpleNamespace(position=position),
            "attitude": None,
            "updated_at": {"position_velocity": float("inf")},
        }
        waypoint = {
            "name": "RWP01", "north_m": 0.0,
            "east_m": 0.0, "down_m": -1.5,
        }
        with patch(
            "src.flight.waypoint_executor.ensure_critical_telemetry_fresh"
        ):
            await waypoint_executor.fly_to_waypoint(
                drone, latest, {}, {}, waypoint, "return_to_start", "return",
                0.6, {"source": "gazebo_lidar_2d"}, object(),
                {"enabled": True, "mode": "active", "risk_level": "danger"},
                {"following_replanned_route": False},
            )
        drone.offboard.set_velocity_ned.assert_awaited()


if __name__ == "__main__":
    unittest.main()
