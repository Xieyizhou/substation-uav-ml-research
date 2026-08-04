import unittest

from src.planner.obstacle_config import build_obstacle_map
from src.vision.collection.layout import (
    build_layout_manifest,
    layout_obstacle_config,
)
from src.vision.collection.route import (
    VisualRoute,
    _cell,
    build_visual_route,
)
from src.vision.contracts.protocol import load_protocol


def _segment_cells(start, destination):
    if start[0] == destination[0]:
        step = 1 if destination[1] >= start[1] else -1
        return {
            (start[0], north)
            for north in range(start[1], destination[1] + step, step)
        }
    if start[1] == destination[1]:
        step = 1 if destination[0] >= start[0] else -1
        return {
            (east, start[1])
            for east in range(start[0], destination[0] + step, step)
        }
    raise AssertionError("simplified A* segment must be axis aligned")


class VisualV2ReturnRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol("v2")
        cls.layout = build_layout_manifest(
            "development-01",
            "development",
            3001,
            cls.protocol["randomization"]["configuration"],
        )

    def route(self):
        return build_visual_route(
            self.layout,
            "transformer_centered_v2",
            "transformer",
        )

    def test_return_path_is_identity_bound_and_ends_at_start(self):
        route = self.route()
        self.assertEqual(route.route_schema_version, 3)
        self.assertEqual(route.return_transit_cells[-1], route.start_cell)
        self.assertEqual(VisualRoute.from_record(route.to_record()), route)

    def test_return_path_is_continuous_and_collision_free(self):
        route = self.route()
        blocked = build_obstacle_map(
            layout_obstacle_config(self.layout)
        )["inflated_blocking_cells"]
        final = route.waypoints[-1]
        current = _cell(final.east_m, final.north_m)
        visited = set()
        for destination in route.return_transit_cells:
            visited.update(_segment_cells(current, destination))
            current = destination
        self.assertFalse(visited & blocked)
        self.assertEqual(current, self.layout.start_cell)


if __name__ == "__main__":
    unittest.main()
