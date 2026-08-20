import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.maps.sandbox_contracts import (
    SandboxMap, SandboxMapObject, SandboxMission,
)
from src.planner.planning_artifacts import (
    materialize_height_layer_plan, materialize_semantic_inspection,
)


def map_value():
    return SandboxMap(
        "artifact_map", "Artifact map", 30, 30, 2.5, 2.5, 0,
        (SandboxMapObject(
            "transformer_a", "transformer", 15, 15, 3, 3, 3,
            label_role="target",
        ),),
        (SandboxMission(
            "round_trip_a", "round_trip", goal_east_m=25, goal_north_m=25,
        ),),
    )


class PlanningArtifactTests(unittest.TestCase):
    def test_materializes_identity_bound_semantic_plan(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_path, observation_path, output = (
                root / "map.json", root / "observation.json", root / "plan.json"
            )
            map_path.write_text(json.dumps(map_value().to_record()))
            observation_path.write_text(json.dumps({
                "observation": {
                    "tracking_id": "temporal-1", "class_name": "transformer",
                    "confidence": 0.9, "bbox": [310, 230, 330, 250],
                    "held": False,
                },
                "depth_m": 12.5,
                "camera_intrinsics": {
                    "width_px": 640, "height_px": 480, "fx_px": 500,
                    "fy_px": 500, "cx_px": 320, "cy_px": 240,
                },
                "vehicle_pose": {
                    "east_m": 15, "north_m": 2.5, "altitude_m": 1.5,
                    "yaw_deg": 0,
                },
            }))
            result = materialize_semantic_inspection(
                map_path, observation_path, output
            )
            self.assertTrue(result["route_quality"]["accepted"])
            self.assertEqual(result["matched_object_id"], "transformer_a")
            self.assertTrue(output.is_file())

    def test_materializes_deterministic_height_layer_plan(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_path = root / "map.json"
            map_path.write_text(json.dumps(map_value().to_record()))
            first = materialize_height_layer_plan(
                map_path, "round_trip_a", root / "first.json"
            )
            second = materialize_height_layer_plan(
                map_path, "round_trip_a", root / "second.json"
            )
            self.assertEqual(
                first["height_layer_plan_identity_sha256"],
                second["height_layer_plan_identity_sha256"],
            )
            self.assertGreater(len(first["nodes"]), 1)


if __name__ == "__main__":
    unittest.main()
