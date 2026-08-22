import base64
from pathlib import Path
import struct
import tempfile
import unittest

from src.sensors.gazebo_depth_codec import parse_gazebo_depth_message
from src.sensors.rgb_depth_sync import RgbDepthSynchronizer, depth_for_bbox


def message(values, width=4, height=4, timestamp=1.0):
    raw = struct.pack(f"<{len(values)}f", *values)
    seconds = int(timestamp); nanos = round((timestamp - seconds) * 1_000_000_000)
    return {"width": width, "height": height, "step": width * 4,
            "pixel_format_type": "R_FLOAT32", "data": base64.b64encode(raw).decode(),
            "header": {"stamp": {"sec": seconds, "nsec": nanos}}}


class RgbDepthTests(unittest.TestCase):
    def test_decode_pair_scale_and_median(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = parse_gazebo_depth_message(message([5.0] * 100, 10, 10), topic="depth",
                staging_directory=directory, receive_index=1, received_monotonic=2)
            self.assertEqual(frame.valid_depth_count, 100)
            self.assertEqual(depth_for_bbox(frame, (20, 20), (0, 0, 20, 20), directory), 5)

    def test_invalid_depth_and_minimum_pixels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = parse_gazebo_depth_message(message([float("nan")] * 16), topic="depth",
                staging_directory=directory, receive_index=1, received_monotonic=2)
            self.assertIsNone(depth_for_bbox(frame, (8, 8), (0, 0, 8, 8), directory))

    def test_pairing_boundary_and_stale_rgb_rejection(self):
        frame = lambda timestamp: type("Frame", (), {"capture_timestamp": timestamp})()
        sync = RgbDepthSynchronizer(33.334)
        self.assertEqual(sync.push_rgb(frame(1.0)), [])
        self.assertEqual(len(sync.push_depth(frame(1.033334))), 1)
        sync.push_rgb(frame(2.0)); sync.push_depth(frame(2.1))
        self.assertEqual(sync.dropped_rgb_count, 1)
        self.assertEqual(sync.receipt()["pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
