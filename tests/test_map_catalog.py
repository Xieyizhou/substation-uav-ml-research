import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from src.maps.map_catalog import (
    CATALOG_PATH,
    current_map,
    field_value,
    list_maps,
    project_path,
    select_map,
    selected_map_id,
)
from src.planner.astar_grid import astar
from src.planner.obstacle_config import (
    build_obstacle_map,
    get_start_goal,
    validate_obstacle_config,
)


class MapCatalogTests(unittest.TestCase):
    def test_catalog_covers_simple_through_extreme(self):
        entries = list_maps()
        self.assertEqual(
            [entry["id"] for entry in entries],
            ["training", "simple", "medium", "complex", "extreme"],
        )
        self.assertEqual([entry["difficulty"] for entry in entries], [1, 2, 3, 4, 5])

    def test_selection_defaults_and_persists_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "selected_map.json"
            self.assertEqual(selected_map_id(CATALOG_PATH, state_path), "simple")
            selected = select_map("complex", CATALOG_PATH, state_path)
            self.assertEqual(selected["id"], "complex")
            self.assertEqual(current_map(CATALOG_PATH, state_path)["id"], "complex")
            self.assertFalse(state_path.with_suffix(".json.tmp").exists())

    def test_field_output_is_shell_friendly(self):
        entry = next(entry for entry in list_maps() if entry["id"] == "extreme")
        self.assertEqual(field_value(entry, "spawn_pose"), "-16,-16,0,0,0,0")


class MapAlignmentAndPlanningTests(unittest.TestCase):
    def test_every_map_is_aligned_and_astar_reachable(self):
        for entry in list_maps():
            with self.subTest(map_id=entry["id"]):
                config_path = project_path(entry["obstacle_config"])
                world_path = project_path(entry["world_file"])
                config = json.loads(config_path.read_text())

                origin = [float(value) for value in config["gazebo_world_origin_m"]]
                self.assertEqual(entry["spawn_pose"][:3], origin)

                world = ET.parse(world_path).getroot().find("world")
                self.assertIsNotNone(world)
                self.assertEqual(world.attrib["name"], entry["world_name"])
                map_model = world.find("./model[@name='substation_map']")
                self.assertIsNotNone(map_model)
                map_pose = [float(value) for value in map_model.findtext("pose").split()[:3]]
                self.assertEqual(map_pose, origin)
                self.assertIsNotNone(map_model.find("./model[@name='substation_floor']"))
                self.assertIsNotNone(map_model.find("./model[@name='substation_floor_grid']"))

                warnings = validate_obstacle_config(config)
                self.assertEqual(warnings, [])
                start, goal = get_start_goal(config)
                obstacle_map = build_obstacle_map(
                    config,
                    start_cell=start,
                    goal_cell=goal,
                )
                route = astar(
                    start=start,
                    goal=goal,
                    obstacles=obstacle_map["inflated_blocking_cells"],
                    width=int(config["width"]),
                    height=int(config["height"]),
                )
                self.assertEqual(route[0], start)
                self.assertEqual(route[-1], goal)
                border_cells = [
                    cell
                    for cell in route
                    if cell[0] in {0, int(config["width"]) - 1}
                    or cell[1] in {0, int(config["height"]) - 1}
                ]
                internal_cells = [
                    cell
                    for cell in route
                    if 2 <= cell[0] <= int(config["width"]) - 3
                    and 2 <= cell[1] <= int(config["height"]) - 3
                ]
                self.assertLessEqual(len(border_cells) / len(route), 0.35)
                self.assertGreaterEqual(len(internal_cells) / len(route), 0.25)


if __name__ == "__main__":
    unittest.main()
