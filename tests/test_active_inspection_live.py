import unittest
from dataclasses import dataclass

from src.vision.active_inspection_live import ActiveInspectionLiveBridge


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    tracking_id: str


class Detector:
    def detect(self, image, **_kwargs):
        return [Detection("transformer", .9, (10, 20, 30, 40), "raw-1")]


class ActiveInspectionLiveBridgeTests(unittest.TestCase):
    def bridge(self, depth):
        def temporal(rows, _timestamp):
            return [{**rows[0], "tracking_id": "temporal-1", "stable": True}]
        return ActiveInspectionLiveBridge(
            Detector(), temporal, lambda _frame, _row: depth,
            lambda _frame: "image", {"fx": 500, "cx": 320},
        )

    def test_stable_detection_with_depth_becomes_planner_record(self):
        frame = type("Frame", (), {"capture_timestamp": 12.5, "frame_id": "rgb-1"})()
        record = self.bridge(7.0).record(
            frame, {"east_m": 1, "north_m": 2, "altitude_m": 3, "yaw_deg": 0},
            elapsed_s=4,
        )
        self.assertEqual(record["trigger"], "stable_detection")
        self.assertEqual(record["observations"][0]["tracking_id"], "temporal-1")
        self.assertEqual(record["observations"][0]["depth_m"], 7)

    def test_missing_or_invalid_depth_cannot_register(self):
        frame = type("Frame", (), {"capture_timestamp": 1.0})()
        pose = {"east_m": 0, "north_m": 0, "altitude_m": 1, "yaw_deg": 0}
        self.assertIsNone(self.bridge(None).record(frame, pose, elapsed_s=1))
        self.assertIsNone(self.bridge(101).record(frame, pose, elapsed_s=1))

    def test_invalid_intrinsics_are_rejected(self):
        with self.assertRaises(ValueError):
            ActiveInspectionLiveBridge(Detector(), lambda *_: [], lambda *_: 1, lambda x: x, {"fx": 0, "cx": 0})


if __name__ == "__main__":
    unittest.main()
