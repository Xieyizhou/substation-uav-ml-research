import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.sensors.gazebo_rgbd_native import NativeGazeboRgbDepthSource


class FakeProcess:
    def __init__(self, rows):
        self.stdout = asyncio.StreamReader()
        for row in rows:
            self.stdout.feed_data(json.dumps(row).encode() + b"\n")
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()
        self.returncode = None

    def terminate(self):
        self.returncode = 0

    async def communicate(self):
        return b"", b""


class NativeRgbDepthSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_metadata_stream_yields_paired_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {
                "pair_count": 6,
                "rgb_frame_count": 6,
                "depth_frame_count": 6,
                "dropped_rgb_count": 0,
                "dropped_depth_count": 0,
                "rgb_pairing_success_rate": 1,
                "depth_pairing_success_rate": 1,
                "skew_ms": 0,
                "skew_p50_ms": 0,
                "skew_p95_ms": 0,
                "skew_max_ms": 0,
                "rgb_timestamp": 1,
                "depth_timestamp": 1,
                "rgb_width": 1920,
                "rgb_height": 1080,
                "depth_width": 640,
                "depth_height": 360,
                "rgb_path": "pair-6.ppm",
                "depth_path": "pair-6.depth",
            }
            source = NativeGazeboRgbDepthSource(directory)
            source.process = FakeProcess([row])
            rgb, depth, skew = await anext(source.events(timeout_s=.1))
            self.assertEqual((rgb.width, rgb.height), (1920, 1080))
            self.assertEqual((depth.width, depth.height), (640, 360))
            self.assertEqual(skew, 0)
            self.assertEqual(source.receipt()["rgb_pairing_success_rate"], 1)

    @patch(
        "src.sensors.gazebo_rgbd_native.compile_native_bridge",
        new_callable=Mock,
    )
    async def test_start_uses_native_binary(self, _compile):
        with tempfile.TemporaryDirectory() as directory:
            process = FakeProcess([])
            with patch(
                "src.sensors.gazebo_rgbd_native.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ) as create:
                source = NativeGazeboRgbDepthSource(Path(directory), emit_stride=6)
                await source.start()
                self.assertEqual(create.await_args.args[0], str(Path("/tmp/substation-uav-gz-rgbd-bridge")))


if __name__ == "__main__":
    unittest.main()
