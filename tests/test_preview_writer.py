import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.flight.preview_writer import save_preview


class PreviewWriterTests(unittest.TestCase):
    def test_save_preview_writes_complete_json_and_png(self):
        planner = {
            "map_name": "synthetic_map",
            "target_id": "center",
            "target_display_name": "Center",
            "gazebo_world_origin_m": [-2, -2, 0],
            "obstacle_config_path": None,
            "width": 3,
            "height": 3,
            "start": (0, 0),
            "goal": (2, 2),
            "resolution_m": 1.0,
            "altitude_m": 1.5,
            "vertical_safety_margin_m": 0.3,
            "horizontal_inflation_cells": 1,
            "blocking_obstacle_names": [],
            "nonblocking_obstacle_names": [],
            "obstacles": set(),
            "raw_obstacle_cells": set(),
            "raw_blocking_cells": set(),
            "inflated_blocking_cells": set(),
            "blocking_obstacle_cells": set(),
            "inflated_obstacle_cells": set(),
            "raw_obstacle_cell_to_name": {},
            "inflated_obstacle_cell_to_name": {},
            "raw_obstacle_cell_count": 0,
            "raw_blocking_cell_count": 0,
            "obstacle_cell_count": 0,
            "inflated_obstacle_cell_count": 0,
            "validation_warnings": [],
        }
        path = [(0, 0), (1, 1), (2, 2)]
        waypoints = [
            {"name": "WP01", "north_m": 0.5, "east_m": 0.5, "down_m": -1.5},
            {"name": "WP02", "north_m": 2.5, "east_m": 2.5, "down_m": -1.5},
        ]
        args = SimpleNamespace(return_home=True, allow_diagonal=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_preview(
                path,
                [(0, 0), (2, 2)],
                waypoints,
                args,
                planner,
                output,
                "astar_grid",
                str,
                lambda items: list(reversed(items)),
            )
            payload = json.loads((output / "path_preview.json").read_text())
            self.assertTrue((output / "grid_path.png").is_file())
            self.assertGreater((output / "grid_path.png").stat().st_size, 0)

        self.assertEqual(payload["map_name"], "synthetic_map")
        self.assertEqual(payload["target_id"], "center")
        self.assertEqual(payload["goal_cell"], [2, 2])
        self.assertTrue(payload["return_home_enabled"])
        self.assertEqual(payload["return_grid_path"], [[2, 2], [1, 1], [0, 0]])


if __name__ == "__main__":
    unittest.main()
