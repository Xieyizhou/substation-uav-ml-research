import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.flight import flight_config
from src.maps.map_catalog import list_maps, project_path
from src.maps.target_catalog import (
    current_target,
    select_target,
    selected_target_for_config,
    targets_for_map,
)
from src.planner.astar_grid import astar
from src.planner.obstacle_config import (
    adjacent_cells,
    build_obstacle_map,
    get_start_goal,
    load_obstacle_config,
)


class TargetCatalogTests(unittest.TestCase):
    def test_every_map_has_five_named_targets(self):
        expected = ["top_right", "center", "left", "bottom", "right"]
        for entry in list_maps():
            with self.subTest(map_id=entry["id"]):
                self.assertEqual(
                    [target["id"] for target in targets_for_map(entry)],
                    expected,
                )

    def test_selections_are_saved_independently_per_map(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "targets.json"
            training, medium = list_maps()[0], list_maps()[2]
            select_target("center", training["id"], state_path)
            select_target("left", medium["id"], state_path)
            self.assertEqual(current_target(training, state_path)["id"], "center")
            self.assertEqual(current_target(medium, state_path)["id"], "left")
            self.assertFalse(state_path.with_suffix(".json.tmp").exists())

    def test_config_path_resolves_its_saved_target(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "targets.json"
            entry = list_maps()[1]
            select_target("bottom", entry["id"], state_path)
            target = selected_target_for_config(
                project_path(entry["obstacle_config"]),
                state_path,
            )
            self.assertEqual(target["id"], "bottom")


class TargetPlanningTests(unittest.TestCase):
    def test_every_target_is_safe_and_reachable(self):
        for entry in list_maps():
            config = load_obstacle_config(project_path(entry["obstacle_config"]))
            start, _ = get_start_goal(config)
            obstacle_map = build_obstacle_map(
                config,
                start_cell=start,
                goal_cell=start,
            )
            blocked = obstacle_map["inflated_blocking_cells"]
            raw = obstacle_map["raw_obstacle_cells"]
            for target in targets_for_map(entry):
                with self.subTest(map_id=entry["id"], target=target["id"]):
                    goal = tuple(target["cell"])
                    self.assertNotIn(goal, raw)
                    self.assertNotIn(goal, blocked)
                    self.assertFalse(
                        adjacent_cells(
                            goal,
                            int(config["width"]),
                            int(config["height"]),
                        )
                        & blocked
                    )
                    self.assertGreaterEqual(goal[0], 1)
                    self.assertGreaterEqual(goal[1], 1)
                    self.assertLess(goal[0], int(config["width"]) - 1)
                    self.assertLess(goal[1], int(config["height"]) - 1)
                    route = astar(
                        start,
                        goal,
                        blocked,
                        int(config["width"]),
                        int(config["height"]),
                    )
                    self.assertEqual(route[-1], goal)

    def test_flight_config_applies_selected_target(self):
        entry = list_maps()[0]
        target = next(item for item in entry["targets"] if item["id"] == "center")
        args = flight_config.parse_args(
            ["--dry-run", "--obstacle-config", entry["obstacle_config"]]
        )
        with (
            patch.object(
                flight_config,
                "selected_target_for_config_path",
                return_value=target,
            ),
            redirect_stdout(StringIO()),
        ):
            planner = flight_config.load_planner_config(args)
        self.assertEqual(planner["goal"], tuple(target["cell"]))
        self.assertEqual(planner["target_id"], "center")


if __name__ == "__main__":
    unittest.main()
