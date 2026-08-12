import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.flight.replanning_controller import (
    attempt_local_replan,
    empty_replan_state,
    route_allows_local_replan,
)


class ReplanningControllerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
