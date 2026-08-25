import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

from src.flight.active_rgbd_runtime import ActiveRgbdRuntime, runtime_map_name
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


class ActiveRgbdRuntimeFeedbackTests(unittest.TestCase):
    def test_runtime_map_name_uses_obstacle_config(self):
        args = SimpleNamespace(obstacle_config="config/substation_obstacles.json")
        self.assertEqual(runtime_map_name(args, {}), "simple")

    def runtime(self, event):
        runtime = object.__new__(ActiveRgbdRuntime)
        runtime.replan_config = {
            "semantic_feedback": [{
                "event": event,
                "candidate_id": "target-1",
                "decision_id": "decision-1",
                "route_identity": "route-1",
                "timestamp_s": 1.0,
            }]
        }
        runtime.replan_state = {}
        runtime.phase_state = {}
        runtime.planner = Mock()
        runtime.planner.tracks = {
            "target-1": SimpleNamespace(source_tracking_ids={"tracking-1"})
        }
        runtime.planner.decisions = []
        runtime._sync_planner_busy = Mock()
        runtime._append_new_decisions = Mock()
        runtime._sync_terminal_state = Mock()
        runtime._elapsed = Mock(return_value=1.0)
        return runtime

    def test_truncated_waypoint_feedback_does_not_complete_target(self):
        runtime = self.runtime("waypoint_reached")
        runtime._drain_feedback(1.0, {})
        record = runtime.planner.process.call_args.args[0]
        self.assertEqual(record["completed_tracking_ids"], [])
        self.assertEqual(record["completed_target_ids"], [])

    def test_completed_feedback_carries_target_and_tracking_ids(self):
        runtime = self.runtime("target_completed")
        runtime._drain_feedback(1.0, {})
        record = runtime.planner.process.call_args.args[0]
        self.assertEqual(record["completed_tracking_ids"], ["tracking-1"])
        self.assertEqual(record["completed_target_ids"], ["target-1"])


if __name__ == "__main__":
    unittest.main()
