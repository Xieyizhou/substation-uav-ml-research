import base64
import json
from types import SimpleNamespace
import unittest

import numpy as np

from src.flight.sim_pose_sync import SimPoseBuffer, MavsdkSimPoseSource
from src.sensors.gazebo_depth_memory import MemoryDepth, decode_depth, aligned_depth, bbox_depth
from src.sensors.gazebo_visual_transport import GAZEBO_SIM_CLOCK


def add(buffer, timestamp, received=100, source=1):
    buffer.add("LOCAL_POSITION_NED", dict(time_boot_ms=timestamp*1000, x=timestamp, y=2, z=-3, vx=.1, vy=0, vz=0),
               received=received, system_id=source, component_id=1)
    buffer.add("ATTITUDE", dict(time_boot_ms=timestamp*1000, roll=0, pitch=0, yaw=.5),
               received=received, system_id=source, component_id=1)


class SimPoseTests(unittest.TestCase):
    def test_matches_capture_time_instead_of_latest_pose_and_bounds_memory(self):
        buffer = SimPoseBuffer(capacity=3)
        for stamp in (9.98, 10, 10.04, 10.08):
            add(buffer, stamp)
        matched = buffer.match(10.005, GAZEBO_SIM_CLOCK, now=100.1)
        self.assertEqual(matched["position_timestamp"], 10)
        self.assertEqual(matched["pose"]["north_m"], 10)
        self.assertEqual(matched["pose"]["altitude_m"], 3)
        self.assertEqual(len(buffer.position), 3)

    def test_rejects_clock_mismatch_skew_and_delayed_images(self):
        buffer = SimPoseBuffer()
        add(buffer, 10)
        for stamp, clock in ((10, "host_monotonic"), (10.1, GAZEBO_SIM_CLOCK), (9.7, GAZEBO_SIM_CLOCK)):
            with self.assertRaises(ValueError):
                buffer.match(stamp, clock, now=100.1)
        with self.assertRaisesRegex(ValueError, "stale"):
            buffer.match(10, GAZEBO_SIM_CLOCK, now=101)
        with self.assertRaises(ValueError):
            buffer.match(float("nan"), GAZEBO_SIM_CLOCK, now=100)
        with self.assertRaisesRegex(ValueError, "explicit Gazebo"):
            MavsdkSimPoseSource(None)

    def test_duplicate_does_not_refresh_pose_and_clock_reset_latches_fault(self):
        buffer = SimPoseBuffer()
        add(buffer, 10, 100)
        add(buffer, 10, 101)
        with self.assertRaisesRegex(ValueError, "stale"):
            buffer.match(10, GAZEBO_SIM_CLOCK, now=101)
        with self.assertRaisesRegex(ValueError, "reversed"):
            add(buffer, 9)
        with self.assertRaisesRegex(ValueError, "reversed"):
            buffer.match(10, GAZEBO_SIM_CLOCK, now=100)

    def test_source_switch_and_nonfinite_pose_are_rejected(self):
        buffer = SimPoseBuffer()
        add(buffer, 10)
        with self.assertRaisesRegex(ValueError, "source changed"):
            add(buffer, 10.01, source=2)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            add(SimPoseBuffer(), float("inf"))


class DepthMemoryTests(unittest.TestCase):
    def test_decode_stride_without_disk_payload(self):
        array = np.full((10, 12), 5, dtype="<f4")
        array[:, 10:] = 99
        message = dict(width=10, height=10, step=48, pixel_format_type="R_FLOAT32",
                       header={"stamp": {"sec": 10, "nsec": 0}}, data=base64.b64encode(array.tobytes()).decode())
        depth = decode_depth(SimpleNamespace(line=json.dumps(message), received=100))
        self.assertEqual(depth.array.shape, (10, 10))
        self.assertFalse(depth.array.flags.writeable)
        self.assertEqual(bbox_depth(depth, (100, 100), [1, 1, 99, 99]), 5)
        message["step"] = 41
        with self.assertRaisesRegex(ValueError, "layout"):
            decode_depth(SimpleNamespace(line=json.dumps(message), received=100))

    def test_inference_delay_skew_and_nonfinite_timing_rejected(self):
        depth = MemoryDepth(10, 100, np.ones((10, 10)), "sha")
        frame = SimpleNamespace(capture_timestamp=10, receive_monotonic_timestamp=100, capture_clock_domain=GAZEBO_SIM_CLOCK)
        self.assertIs(aligned_depth(frame, depth, now=100.1), depth)
        with self.assertRaisesRegex(ValueError, "receive age"):
            aligned_depth(frame, depth, now=100.6)
        frame.capture_timestamp = 10.04
        with self.assertRaisesRegex(ValueError, "capture skew"):
            aligned_depth(frame, depth, now=100.1)
        frame.capture_timestamp = float("nan")
        with self.assertRaisesRegex(ValueError, "capture time"):
            aligned_depth(frame, depth, now=100.1)

    def test_truncation_mixed_surfaces_and_sparse_depth_rejected(self):
        array = np.full((20, 20), 5.)
        depth = MemoryDepth(10, 100, array, "sha")
        self.assertIsNone(bbox_depth(depth, (20, 20), [0, 0, 20, 20]))
        array[:, 10:] = 15
        self.assertIsNone(bbox_depth(depth, (20, 20), [1, 1, 19, 19]))
        array[:, 10:] = float("nan")
        self.assertIsNone(bbox_depth(depth, (20, 20), [1, 1, 19, 19]))


if __name__ == "__main__":
    unittest.main()
