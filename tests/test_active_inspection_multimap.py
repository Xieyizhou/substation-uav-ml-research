import json
import tempfile
import unittest
from pathlib import Path

from scripts.flight.aggregate_active_inspection_matrix import (
    identity,
    truth_from_obstacles,
    verified_class_accuracy,
)


class ActiveInspectionMultimapTests(unittest.TestCase):
    def test_truth_adapter_normalizes_classes_and_centers(self):
        truth = truth_from_obstacles({
            "resolution_m": 2,
            "obstacles": [
                {"name": "c1", "visual_category": "capacitor", "x_min": 1, "x_max": 3, "y_min": 2, "y_max": 4},
                {"name": "wall", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
            ],
        })
        self.assertEqual(truth, [{
            "id": "c1",
            "class_name": "capacitor_bank",
            "east_min_m": 2.0,
            "east_max_m": 6.0,
            "north_min_m": 4.0,
            "north_max_m": 8.0,
            "east_m": 4.0,
            "north_m": 6.0,
        }])

    def test_identity_is_deterministic(self):
        self.assertEqual(identity({"b": 2, "a": 1}), identity({"a": 1, "b": 2}))

    def test_class_accuracy_requires_spatially_verified_class(self):
        observed = {"transformer", "switchgear", "capacitor_bank"}
        self.assertEqual(
            verified_class_accuracy(
                ["transformer", "transformer", "switchgear", "capacitor_bank"],
                observed,
            ),
            1.0,
        )
        self.assertEqual(
            verified_class_accuracy(["transformer", "switchgear"], observed),
            2 / 3,
        )

    def test_manifest_has_three_maps_and_twenty_seven_runs(self):
        manifest = json.loads(Path("config/perception/active_inspection_multimap_matrix_v1.json").read_text())
        self.assertEqual([row["map_id"] for row in manifest["maps"]], ["simple", "medium", "complex"])
        self.assertEqual(sum(len(ids) for row in manifest["maps"] for ids in row["runs"].values()), 27)
        self.assertEqual(manifest["maps"][0]["expected_classes"], ["transformer", "switchgear"])
        self.assertEqual(len(manifest["maps"][1]["expected_classes"]), 3)
        self.assertEqual(len(manifest["maps"][2]["expected_classes"]), 4)


if __name__ == "__main__":
    unittest.main()
