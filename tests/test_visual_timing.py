import hashlib
import time
import unittest

from src.ml import EQUIPMENT_CLASSES
from src.ml.equipment_detector import EquipmentDetector
from src.sensors.types import (
    CameraFrame,
    LOCAL_MONOTONIC_CLOCK,
    VisualTiming,
    VisualTimingContext,
)
from src.sensors.visual_timing import summarize_visual_timings


class _Values:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        value = self.values[index]
        return _Values(value) if isinstance(value, list) else value


class _Boxes:
    def __init__(self):
        self.cls = _Values([0, 1, 2, 3])
        self.xyxy = _Values(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        )
        self.conf = _Values([0.9, 0.8, 0.7, 0.6])
        self.id = None


class _Result:
    names = dict(enumerate(EQUIPMENT_CLASSES))
    boxes = _Boxes()
    speed = {"preprocess": 1.25, "inference": 2.5, "postprocess": 0.75}


class _Model:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_Result()]


class _Image:
    shape = (480, 640, 3)


def _detector():
    detector = EquipmentDetector.__new__(EquipmentDetector)
    detector.model = _Model()
    detector.confidence = 0.25
    detector.device = "cpu"
    detector.model_id = "mock-equipment"
    detector.image_size = 640
    detector.runtime_backend = "mock-ultralytics"
    return detector


def _frame(clock_domain="simulator"):
    now = time.monotonic()
    payload = b"not-loaded-by-detector"
    return CameraFrame(
        frame_id="camera-1",
        source_id="test-camera",
        sequence_number=1,
        capture_timestamp=now - 0.01 if clock_domain == LOCAL_MONOTONIC_CLOCK else 1.0,
        capture_clock_domain=clock_domain,
        receive_monotonic_timestamp=now - 0.005,
        width=640,
        height=480,
        payload_format="png",
        pixel_format="rgb8",
        payload_relative_path="frames/000000001.png",
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


class VisualTimingContractTests(unittest.TestCase):
    def test_unavailable_stages_remain_null(self):
        timing = VisualTiming()
        self.assertIsNone(timing.decode_ms)
        self.assertIsNone(timing.inference_ms)
        self.assertIsNone(timing.end_to_end_ms)
        self.assertIsNone(timing.deadline_missed)

    def test_summary_excludes_unavailable_values(self):
        summary = summarize_visual_timings(
            [
                VisualTiming(backend_call_ms=1.0),
                VisualTiming(backend_call_ms=None),
                VisualTiming(backend_call_ms=3.0),
            ]
        )
        backend = summary["stages"]["backend_call_ms"]
        self.assertEqual(backend["count"], 2)
        self.assertEqual(backend["mean_ms"], 2.0)
        self.assertIsNone(summary["stages"]["decode_ms"]["p95_ms"])

    def test_incompatible_clock_domains_do_not_fabricate_latency(self):
        frame = _frame("simulator")
        self.assertIsNone(frame.capture_to_receive_ms())


class EquipmentDetectorTimingTests(unittest.TestCase):
    def test_detect_remains_compatible_and_uses_locked_default_size(self):
        detector = _detector()
        detections = detector.detect(_Image(), timestamp_s=7.0, frame_id="camera")
        self.assertEqual(tuple(item.class_name for item in detections), EQUIPMENT_CLASSES)
        self.assertTrue(all(item.timestamp_s == 7.0 for item in detections))
        self.assertEqual(detector.model.calls[0]["imgsz"], 640)

    def test_detect_with_metrics_returns_equivalent_detections_and_backend_stages(self):
        detector = _detector()
        expected = detector.detect(_Image(), timestamp_s=7.0, frame_id="camera")
        result = detector.detect_with_metrics(
            _Image(),
            timestamp_s=7.0,
            frame_id="camera",
            timing_context=VisualTimingContext(frame=_frame("simulator")),
        )
        self.assertEqual(result.detections, expected)
        self.assertGreaterEqual(result.timing.backend_call_ms, 0.0)
        self.assertGreaterEqual(result.timing.decision_finalize_ms, 0.0)
        self.assertEqual(result.timing.preprocess_ms, 1.25)
        self.assertEqual(result.timing.inference_ms, 2.5)
        self.assertEqual(result.timing.postprocess_ms, 0.75)
        self.assertIsNone(result.timing.decode_ms)
        self.assertIsNone(result.timing.capture_to_receive_ms)
        self.assertIsNone(result.timing.end_to_end_ms)
        self.assertEqual(result.timing.detection_count, 4)
        self.assertEqual(result.timing.input_width, 640)
        self.assertEqual(result.timing.input_height, 480)
        self.assertEqual(
            result.timing.timing_provenance["inference_ms"], "backend_reported"
        )

    def test_comparable_deadline_is_evaluated(self):
        detector = _detector()
        context = VisualTimingContext(
            frame=_frame(LOCAL_MONOTONIC_CLOCK),
            queue_entered_monotonic_timestamp=time.monotonic() - 0.02,
            queue_depth=3,
            deadline_ms=0.001,
        )
        result = detector.detect_with_metrics(_Image(), timing_context=context)
        self.assertIsNotNone(result.timing.capture_to_receive_ms)
        self.assertIsNotNone(result.timing.queue_wait_ms)
        self.assertIsNotNone(result.timing.end_to_end_ms)
        self.assertTrue(result.timing.deadline_missed)
        self.assertEqual(result.timing.queue_depth, 3)


if __name__ == "__main__":
    unittest.main()
