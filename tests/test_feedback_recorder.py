import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from src.sensors.types import CameraFrame
from src.sensors.camera_decoded import decoded_content_sha256
from src.sensors.camera_decoder import decode_camera_payload
from src.vision.collection.feedback_recorder import FeedbackPolicy, FeedbackRecorder, BoundedFeedbackWriter


def event(value=0, timestamp=1.0, received=None):
    rgb = np.full((32, 32, 3), value, dtype=np.uint8)
    rgb.setflags(write=False)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    frame = CameraFrame(str(value), "camera-a", value, timestamp, "simulation", time.monotonic() if received is None else received,
                        32, 32, "raw", "rgb8", f"frames/{digest}.raw", digest)
    return SimpleNamespace(frame=frame, decoded=SimpleNamespace(image=SimpleNamespace(
        array=rgb, decoded_content_sha256=decoded_content_sha256(rgb))))


class FeedbackRecorderTests(unittest.TestCase):
    def test_lossless_identity_resume_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            policy = FeedbackPolicy(max_frames=2)
            writer = FeedbackRecorder(root, "source-hash", policy)
            first = event()
            saved = writer.record(first, [{"class_name": "transformer", "confidence": 0.9}])
            decoded = decode_camera_payload(CameraFrame.from_record(saved["frame"]), root)
            np.testing.assert_array_equal(decoded.image.array, first.decoded.image.array)
            self.assertEqual(writer.record(first, [])["reason"], "duplicate_pixels")
            self.assertEqual(writer.record(event(1, 1.1), [])["reason"], "sample_interval")
            writer.close()
            resumed = FeedbackRecorder(root, "source-hash", policy, resume=True)
            self.assertEqual(resumed.counts["recovered_samples"], 1)
            self.assertTrue(resumed.record(event(2, 2), [])["saved"])
            self.assertEqual(resumed.record(event(3, 3), [])["reason"], "frame_budget")
            summary = resumed.close()
            actual = sum(p.stat().st_size for name in ("frames", "samples") for p in (root/name).iterdir())
            self.assertEqual(summary["sample_bytes"], actual)
            self.assertFalse(summary["training_admitted"])
            self.assertTrue(all(json.loads(p.read_text())["annotation_status"] == "unreviewed" for p in (root/"samples").glob("*.json")))

    def test_stale_budget_and_pixels_rejected_without_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = FeedbackRecorder(Path(temporary)/"collection", "source", FeedbackPolicy(max_sample_bytes=1))
            self.assertEqual(writer.record(event(received=1), [], now=3)["reason"], "stale_or_future_receive_time")
            self.assertEqual(writer.record(event(), [])["reason"], "storage_budget")
            self.assertEqual(writer.close()["sample_bytes"], 0)

    def test_resume_checks_source_hash_and_single_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"collection"
            writer = FeedbackRecorder(root, "source")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                FeedbackRecorder(root, "source", resume=True)
            writer.record(event(), [])
            writer.close()
            with self.assertRaisesRegex(ValueError, "source or policy changed"):
                FeedbackRecorder(root, "another-source", resume=True)
            image = next((root/"frames").glob("*.png"))
            image.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "image identity"):
                FeedbackRecorder(root, "source", resume=True)

    def test_interrupted_payload_is_counted_but_not_admitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"collection"
            writer = FeedbackRecorder(root, "source")
            writer.close()
            (root/"frames").mkdir()
            (root/"frames/orphan.png.partial").write_bytes(b"orphan")
            writer = FeedbackRecorder(root, "source", resume=True)
            self.assertEqual(len(writer.members), 0)
            self.assertEqual(writer.bytes, 6)
            writer.close()

    def test_queue_is_bounded_and_does_not_block_submission(self):
        async def exercise(root):
            writer = BoundedFeedbackWriter(FeedbackRecorder(root, "source", FeedbackPolicy(queue_capacity=2)))
            for value in range(20):
                writer.submit(event(value, value), [])
                self.assertLessEqual(writer.queue.qsize(), 2)
            result = await writer.close()
            self.assertEqual(result["session_counts"]["queue_replaced"], 18)
            self.assertEqual(result["sample_count"], 2)
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(exercise(Path(temporary)/"collection"))

    def test_failed_source_marks_capture_failed_and_stops_source(self):
        from scripts.vision.collect_feedback import observe
        class Source:
            stopped = False
            async def start(self):
                raise RuntimeError("camera disconnected")
            async def stop(self):
                self.stopped = True
        source = Source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"live"
            info = {"onnx_model_sha256": "model", "receipt_identity_sha256": "receipt"}
            with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
                asyncio.run(observe(None, info, root, FeedbackPolicy(), topic="test", seconds=1,
                                    source_factory=lambda *a, **kw: source))
            self.assertTrue(source.stopped)
            self.assertEqual(json.loads((root/"status.json").read_text())["state"], "failed")
            self.assertEqual(json.loads((root/"feedback/summary.json").read_text())["state"], "failed")

    def test_cleanup_failure_does_not_report_complete(self):
        from scripts.vision.collect_feedback import observe
        class Source:
            async def start(self):
                pass
            async def stop(self):
                raise RuntimeError("cleanup failed")
            async def events(self, **kwargs):
                yield SimpleNamespace(received=time.monotonic(), materialize=lambda: event())
                await asyncio.sleep(10)
        model = SimpleNamespace(predict=lambda **kw: [SimpleNamespace(boxes=None)])
        info = {"onnx_model_sha256": "model", "receipt_identity_sha256": "receipt", "imgsz": 32, "threshold": .5}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"live"
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                asyncio.run(observe(model, info, root, FeedbackPolicy(), topic="test", seconds=.05,
                                    source_factory=lambda *a, **kw: Source()))
            self.assertEqual(json.loads((root/"status.json").read_text())["state"], "failed")
            self.assertEqual(json.loads((root/"feedback/summary.json").read_text())["error"], "RuntimeError: cleanup failed")


if __name__ == "__main__":
    unittest.main()
