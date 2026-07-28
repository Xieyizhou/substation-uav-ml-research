import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.flight import flight_config
from src.flight.test_simple_perception import run_smoke
from src.maps.map_catalog import list_maps, project_path


class FlightMapSelectionTests(unittest.TestCase):
    def test_direct_runner_defaults_to_selected_map(self):
        selected_path = Path("/tmp/selected-map.json")
        with patch.object(
            flight_config,
            "selected_obstacle_config_path",
            return_value=selected_path,
        ):
            args = flight_config.parse_args(["--dry-run"])
        self.assertEqual(args.obstacle_config, selected_path)

    def test_explicit_config_overrides_selected_map(self):
        explicit_path = Path("config/custom.json")
        with patch.object(
            flight_config,
            "selected_obstacle_config_path",
            side_effect=AssertionError("selected map should not be resolved"),
        ):
            args = flight_config.parse_args(
                ["--dry-run", "--obstacle-config", str(explicit_path)]
            )
        self.assertEqual(args.obstacle_config, explicit_path)

    def test_built_in_grid_requires_explicit_flag(self):
        with patch.object(
            flight_config,
            "selected_obstacle_config_path",
            side_effect=AssertionError("selected map should not be resolved"),
        ):
            args = flight_config.parse_args(["--dry-run", "--use-built-in-grid"])
        self.assertIsNone(args.obstacle_config)

    def test_perception_smoke_supports_every_catalog_map(self):
        for entry in list_maps():
            with self.subTest(map_id=entry["id"]):
                config_path = project_path(entry["obstacle_config"])
                obstacle_config = json.loads(config_path.read_text())
                with redirect_stdout(StringIO()):
                    run_smoke(obstacle_config)


if __name__ == "__main__":
    unittest.main()
