import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from src.maps.goal_marker import (
    PX4_WORLD_SYSTEMS,
    prepare_world_with_target,
    sync_running_goal_marker,
    target_local_position,
    target_world_position,
)
from src.maps.map_catalog import list_maps, project_path
from src.maps.target_catalog import targets_for_map


class GoalMarkerWorldTests(unittest.TestCase):
    def test_runtime_world_marker_matches_every_target(self):
        with tempfile.TemporaryDirectory() as directory:
            for entry in list_maps():
                source = project_path(entry["world_file"])
                for target in targets_for_map(entry):
                    with self.subTest(map_id=entry["id"], target=target["id"]):
                        output = Path(directory) / f"{entry['id']}_{target['id']}.sdf"
                        prepare_world_with_target(source, output, entry, target)
                        root = ET.parse(output).getroot()
                        marker = next(
                            model
                            for model in root.iter("model")
                            if model.get("name") == "goal_marker"
                        )
                        actual = tuple(float(value) for value in marker.findtext("pose").split()[:3])
                        self.assertEqual(actual, target_local_position(entry, target))
                        plugins = {
                            plugin.get("name"): plugin
                            for plugin in root.find("world").findall("plugin")
                        }
                        self.assertEqual(
                            set(plugins),
                            {name for _, name in PX4_WORLD_SYSTEMS},
                        )
                        self.assertEqual(
                            plugins["gz::sim::systems::Sensors"].findtext(
                                "render_engine"
                            ),
                            "ogre2",
                        )

    def test_world_position_adds_map_origin(self):
        entry = next(item for item in list_maps() if item["id"] == "extreme")
        target = next(item for item in entry["targets"] if item["id"] == "center")
        self.assertEqual(target_local_position(entry, target), (16.5, 18.5, 0.035))
        self.assertEqual(target_world_position(entry, target), (0.5, 2.5, 0.035))


class LiveGoalMarkerTests(unittest.TestCase):
    def test_live_sync_uses_selected_world_position(self):
        entry = next(item for item in list_maps() if item["id"] == "extreme")
        target = next(item for item in entry["targets"] if item["id"] == "center")
        completed = type("Result", (), {"returncode": 0, "stdout": "data: true", "stderr": ""})()
        with (
            patch("src.maps.goal_marker.px4_launcher_pid", return_value=123),
            patch("src.maps.goal_marker.shutil.which", return_value="/usr/bin/gz"),
            patch("src.maps.goal_marker.subprocess.run", return_value=completed) as run,
        ):
            status, _ = sync_running_goal_marker(entry, target)
        self.assertEqual(status, "updated")
        request = run.call_args.args[0][-1]
        self.assertIn("x: 0.5", request)
        self.assertIn("y: 2.5", request)


if __name__ == "__main__":
    unittest.main()
