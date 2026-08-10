import asyncio
import base64
import json
import tempfile
import threading
import unittest
from unittest import mock

from src.sensors.gazebo_camera import (
    GazeboCameraSource,
    parse_gazebo_image_message,
)


def _image_message(timestamp):
    return {
        "header": {"stamp": {"sec": timestamp, "nsec": 0}, "data": []},
        "width": 2,
        "height": 1,
        "step": 6,
        "data": base64.b64encode(b"\0" * 6).decode(),
        "pixel_format_type": "RGB_INT8",
    }


class _LineStream:
    def __init__(self, lines=()):
        self.lines = list(lines)

    async def readline(self):
        if self.lines:
            return self.lines.pop(0)
        await asyncio.Event().wait()

    async def read(self):
        return b""


class _Process:
    def __init__(self, lines):
        self.stdout = _LineStream(lines)
        self.stderr = _LineStream()
        self.returncode = None


class GazeboCameraPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_png_encoding_runs_concurrently_without_reordering(self):
        lines = [
            json.dumps(_image_message(index)).encode() + b"\n"
            for index in (1, 2)
        ]
        barrier = threading.Barrier(2, timeout=1)
        original = parse_gazebo_image_message

        def synchronized_parse(*args, **kwargs):
            barrier.wait()
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            source = GazeboCameraSource(
                directory,
                topic="/research_camera/image",
                encoder_workers=2,
            )
            source._process = _Process(lines)
            events = source.events(timeout_s=1)
            with mock.patch(
                "src.sensors.gazebo_camera.parse_gazebo_image_message",
                side_effect=synchronized_parse,
            ):
                first = await anext(events)
                second = await anext(events)
            await events.aclose()
        self.assertEqual(
            [first.receive_index, second.receive_index],
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
