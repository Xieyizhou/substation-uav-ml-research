import unittest

from src.maps.sandbox_contracts import SandboxMap, SandboxMapObject
from src.planner.semantic_inspection import (
    CameraIntrinsics,
    SemanticEquipmentEstimate,
    VehiclePose,
    localize_equipment,
    plan_semantic_inspection,
)


def semantic_map():
    return SandboxMap(
        "semantic_map", "Semantic map", 30, 30, 2.5, 2.5, 0,
        (SandboxMapObject(
            "transformer_a", "transformer", 15, 15, 3, 3, 3,
            label_role="target",
        ),),
        (),
    )


class SemanticInspectionPlanningTests(unittest.TestCase):
    def test_center_detection_projects_along_vehicle_yaw(self):
        observation = {
            "tracking_id": "temporal-7",
            "class_name": "transformer",
            "confidence": 0.9,
            "bbox": [300, 220, 340, 260],
            "held": False,
        }
        estimate = localize_equipment(
            observation, 10,
            CameraIntrinsics(640, 480, 500, 500, 320, 240),
            VehiclePose(5, 5, 2, 90),
        )
        self.assertAlmostEqual(estimate.east_m, 15)
        self.assertAlmostEqual(estimate.north_m, 5)
        self.assertAlmostEqual(estimate.altitude_m, 2)

    def test_held_observation_cannot_initialize_location(self):
        observation = {
            "tracking_id": "temporal-1", "class_name": "transformer",
            "confidence": 0.8, "bbox": [0, 0, 10, 10], "held": True,
        }
        with self.assertRaisesRegex(ValueError, "held"):
            localize_equipment(
                observation, 3,
                CameraIntrinsics(20, 20, 10, 10, 10, 10),
                VehiclePose(0, 0, 1, 0),
            )

    def test_spatial_estimate_generates_quality_gated_inspection(self):
        estimate = SemanticEquipmentEstimate(
            "transformer", 0.9, 15.5, 14.5, 1.5, 8, "temporal-1"
        )
        plan = plan_semantic_inspection(semantic_map(), estimate)
        self.assertEqual(plan.matched_object_id, "transformer_a")
        self.assertTrue(plan.route_quality["accepted"])
        self.assertEqual(plan.mission.mission_type, "equipment_inspection")

    def test_class_or_location_mismatch_is_rejected(self):
        wrong_class = SemanticEquipmentEstimate(
            "reactor", 0.9, 15, 15, 1, 8, "temporal-1"
        )
        with self.assertRaisesRegex(ValueError, "no target"):
            plan_semantic_inspection(semantic_map(), wrong_class)
        far = SemanticEquipmentEstimate(
            "transformer", 0.9, 25, 25, 1, 8, "temporal-2"
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            plan_semantic_inspection(semantic_map(), far)


if __name__ == "__main__":
    unittest.main()
