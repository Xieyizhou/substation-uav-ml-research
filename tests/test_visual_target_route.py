from copy import deepcopy
import unittest

from src.flight.visual_target_route import StableVisualTarget, plan_visual_standoff
from src.planner.local_frame import LocalFrame


def observation(x=8, confidence=.95):
    return dict(class_name="transformer", confidence=confidence,
                local_position=dict(east_m=x, north_m=0, altitude_m=2))


def confirmed():
    tracker = StableVisualTarget()
    request = None
    for i in range(3):
        request = tracker.update([observation()], {"capture_timestamp": 10+i*.1}, now=100+i*.1)
    return tracker, request


class VisualTargetTests(unittest.TestCase):
    def test_requires_three_fresh_consistent_frames(self):
        tracker = StableVisualTarget()
        self.assertIsNone(tracker.update([observation()], {"capture_timestamp": 10}, now=100))
        self.assertIsNone(tracker.update([observation()], {"capture_timestamp": 10}, now=100.1))
        self.assertIsNone(tracker.update([observation(9)], {"capture_timestamp": 10.2}, now=100.2))
        tracker, request = confirmed()
        self.assertIsNotNone(request)
        self.assertIsNone(tracker.current(101))
        tracker.update([], {"capture_timestamp": 10.3}, now=100.3)
        self.assertIsNone(tracker.current(100.3))

    def test_target_and_registration_determine_changed_safe_goal(self):
        _, request = confirmed()
        plan = plan_visual_standoff(request, now=100.3, frame=LocalFrame(1.5, 1.5), start=(1.5, 1.5), boxes=[])
        self.assertEqual(plan["detected_map_position"], (9.5, 1.5))
        self.assertAlmostEqual(plan["route"][-1][0], 4.5)
        self.assertAlmostEqual(plan["route"][-1][1], 1.5)

    def test_expiry_tamper_and_no_path_rejected(self):
        _, request = confirmed()
        args = dict(frame=LocalFrame(1.5, 1.5), start=(1.5, 1.5), boxes=[])
        with self.assertRaisesRegex(ValueError, "expired"):
            plan_visual_standoff(request, now=101, **args)
        changed = deepcopy(request)
        changed["local_position"]["east_m"] = 20
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            plan_visual_standoff(changed, now=100.3, **args)
        args["boxes"] = [dict(x0=0, x1=20, y0=0, y1=20)]
        with self.assertRaisesRegex(ValueError, "no safe"):
            plan_visual_standoff(request, now=100.3, **args)


if __name__ == "__main__":
    unittest.main()
