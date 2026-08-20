import unittest

import pandas as pd

from src.logging.collision_checks import obstacle_collision_report


class CollisionChecksTest(unittest.TestCase):
    def test_low_obstacle_footprint_is_not_a_collision(self):
        frame = pd.DataFrame(
            [{"elapsed_s": 1.0, "local_east_m": 2.2, "local_north_m": 3.2}]
        )
        obstacle_map = {
            "raw_obstacle_cells": {(2, 3)},
            "raw_blocking_cells": set(),
            "inflated_blocking_cells": set(),
            "raw_obstacle_cell_to_name": {(2, 3): "low_asset"},
            "raw_blocking_cell_to_name": {},
        }

        report = obstacle_collision_report(frame, obstacle_map, 1.0)

        self.assertFalse(report["raw_physical_collision_detected"])
        self.assertEqual(report["raw_obstacle_names_involved"], [])

    def test_altitude_blocking_footprint_remains_a_collision(self):
        frame = pd.DataFrame(
            [{"elapsed_s": 1.0, "local_east_m": 2.2, "local_north_m": 3.2}]
        )
        obstacle_map = {
            "raw_obstacle_cells": {(2, 3)},
            "raw_blocking_cells": {(2, 3)},
            "inflated_blocking_cells": {(2, 3)},
            "raw_blocking_cell_to_name": {(2, 3): "transformer"},
            "inflated_obstacle_cell_to_name": {(2, 3): "transformer"},
        }

        report = obstacle_collision_report(frame, obstacle_map, 1.0)

        self.assertTrue(report["raw_physical_collision_detected"])
        self.assertEqual(report["raw_obstacle_names_involved"], ["transformer"])


if __name__ == "__main__":
    unittest.main()
