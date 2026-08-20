import unittest

from src.maps.sandbox_contracts import SandboxMap, SandboxMapObject
from src.planner.astar_25d import (
    HeightLayerGrid,
    grid_from_sandbox_map,
    plan_height_layer_route,
)


class HeightLayerAStarTests(unittest.TestCase):
    def test_uses_high_layer_to_cross_tall_wall(self):
        wall = SandboxMapObject(
            "wall", "generic_obstacle", 8, 8, 1, 16, 2.0,
        )
        map_value = SandboxMap(
            "height_map", "Height map", 16, 16, 2.5, 2.5, 0,
            (wall,), (),
        )
        grid = grid_from_sandbox_map(
            map_value, (1.5, 3.0), horizontal_inflation_cells=0
        )
        route = plan_height_layer_route(grid, (2, 8, 0), (13, 8, 0))
        self.assertIn(1, {node[2] for node in route.nodes})
        self.assertEqual(route.layer_change_count, 2)
        self.assertEqual(route.vertical_distance_m, 3.0)

    def test_route_is_deterministic(self):
        grid = HeightLayerGrid(5, 5, (1.5, 3.0), frozenset())
        first = plan_height_layer_route(grid, (0, 0, 0), (4, 4, 1))
        second = plan_height_layer_route(grid, (0, 0, 0), (4, 4, 1))
        self.assertEqual(first, second)

    def test_rejects_blocked_endpoint_and_unreachable_grid(self):
        grid = HeightLayerGrid(
            3, 3, (1.0,), frozenset({(1, north, 0) for north in range(3)})
        )
        with self.assertRaisesRegex(ValueError, "No 2.5D path"):
            plan_height_layer_route(grid, (0, 1, 0), (2, 1, 0))
        with self.assertRaisesRegex(ValueError, "start"):
            plan_height_layer_route(grid, (1, 1, 0), (2, 1, 0))

    def test_vertical_clearance_controls_layer_occupancy(self):
        item = SandboxMapObject(
            "cabinet", "cabinet", 8, 8, 2, 2, 2.5,
        )
        map_value = SandboxMap(
            "clearance_map", "Clearance map", 16, 16, 2.5, 2.5, 0,
            (item,), (),
        )
        grid = grid_from_sandbox_map(
            map_value, (2.8, 3.1), horizontal_inflation_cells=0,
            drone_half_height_m=0.2, vertical_clearance_m=0.2,
        )
        self.assertIn((8, 8, 0), grid.blocked)
        self.assertNotIn((8, 8, 1), grid.blocked)


if __name__ == "__main__":
    unittest.main()
