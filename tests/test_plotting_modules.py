import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from src.logging import plotting


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PlottingModuleTests(unittest.TestCase):
    def test_compatibility_module_preserves_public_plot_functions(self):
        expected = {
            "save_collision_zoom_plot",
            "save_detection_count_plot",
            "save_error_plot",
            "save_line_plot",
            "save_perception_risk_timeline",
            "save_perception_timeline",
            "save_target_timeline",
            "save_trajectory_plot",
            "target_sequence",
        }
        self.assertTrue(expected.issubset(set(plotting.__all__)))
        for name in expected:
            self.assertTrue(callable(getattr(plotting, name)))

    def test_trajectory_plot_supports_perception_risk_points(self):
        frame = pd.DataFrame(
            {
                "elapsed_s": [0.0, 1.0, 2.0],
                "local_north_m": [0.0, 0.5, 1.0],
                "local_east_m": [0.0, 0.5, 1.0],
                "target_name": ["WP01", "WP01", "WP02"],
                "target_north_m": [1.0, 1.0, 2.0],
                "target_east_m": [1.0, 1.0, 2.0],
                "target_down_m": [-1.0, -1.0, -1.0],
                "route_direction": ["outbound"] * 3,
                "phase": ["flight"] * 3,
                "perception_enabled": [True] * 3,
                "detected_obstacle": [False, True, True],
                "risk_level": ["clear", "warning", "danger"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "trajectory.png"
            result = plotting.save_trajectory_plot(
                frame,
                output_path,
                PROJECT_ROOT / "synthetic.csv",
                "simple",
                {},
                1.0,
                show_perception_points=True,
            )
            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
