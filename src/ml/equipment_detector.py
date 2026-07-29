"""Four-class YOLO equipment detector with optional ONNX export."""

from __future__ import annotations

from pathlib import Path
import math
import time

from src.ml import EQUIPMENT_CLASSES
from src.sensors.types import (
    EquipmentDetection,
    LOCAL_MONOTONIC_CLOCK,
    VisualDetectionResult,
    VisualTiming,
    VisualTimingContext,
)


class EquipmentDetector:
    def __init__(self, weights, confidence=0.25, device="cpu", image_size=640):
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size <= 0
        ):
            raise ValueError("image_size must be a positive integer")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("YOLO equipment inference requires requirements-ml.txt") from error
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        self.model = YOLO(str(self.weights))
        self.confidence = float(confidence)
        self.device = device
        self.model_id = self.weights.stem
        self.image_size = image_size
        self.runtime_backend = "ultralytics"

    def detect(self, image, *, timestamp_s=None, frame_id="research_rgb"):
        return self.detect_with_metrics(
            image,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
        ).detections

    def detect_with_metrics(
        self,
        image,
        *,
        timestamp_s=None,
        frame_id="research_rgb",
        timing_context=None,
    ):
        if timing_context is None:
            timing_context = VisualTimingContext()
        if not isinstance(timing_context, VisualTimingContext):
            raise TypeError("timing_context must be a VisualTimingContext")
        if timestamp_s is None and timing_context.frame is not None:
            timestamp_s = timing_context.frame.capture_timestamp
        timestamp_s = time.time() if timestamp_s is None else timestamp_s
        backend_started_s = time.monotonic_ns() / 1_000_000_000
        results = self.model.predict(
            source=image,
            imgsz=getattr(self, "image_size", 640),
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        backend_finished_s = time.monotonic_ns() / 1_000_000_000
        detections = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for index, class_index in enumerate(boxes.cls.tolist()):
                class_name = names[int(class_index)]
                if class_name not in EQUIPMENT_CLASSES:
                    continue
                xyxy = tuple(float(value) for value in boxes.xyxy[index].tolist())
                detections.append(
                    EquipmentDetection(
                        class_name=class_name,
                        confidence=float(boxes.conf[index]),
                        bbox_xyxy=xyxy,
                        timestamp_s=timestamp_s,
                        frame_id=frame_id,
                        tracking_id=(
                            str(int(boxes.id[index]))
                            if boxes.id is not None
                            else None
                        ),
                    )
                )
        finalized_s = time.monotonic_ns() / 1_000_000_000
        frame = timing_context.frame
        capture_to_receive_ms = (
            frame.capture_to_receive_ms() if frame is not None else None
        )
        queue_wait_ms = None
        if timing_context.queue_entered_monotonic_timestamp is not None:
            queue_wait_s = (
                backend_started_s
                - timing_context.queue_entered_monotonic_timestamp
            )
            if queue_wait_s >= 0:
                queue_wait_ms = queue_wait_s * 1000.0
        end_to_end_ms = None
        if (
            frame is not None
            and frame.capture_clock_domain == LOCAL_MONOTONIC_CLOCK
            and finalized_s >= frame.capture_timestamp
        ):
            end_to_end_ms = (finalized_s - frame.capture_timestamp) * 1000.0
        deadline_duration_ms = end_to_end_ms
        if (
            deadline_duration_ms is None
            and timing_context.queue_entered_monotonic_timestamp is not None
            and finalized_s >= timing_context.queue_entered_monotonic_timestamp
        ):
            deadline_duration_ms = (
                finalized_s
                - timing_context.queue_entered_monotonic_timestamp
            ) * 1000.0
        deadline_missed = None
        if (
            timing_context.deadline_ms is not None
            and deadline_duration_ms is not None
        ):
            deadline_missed = deadline_duration_ms > timing_context.deadline_ms
        backend_stage_timings = _backend_stage_timings(results)
        input_width, input_height = _input_dimensions(image, frame)
        provenance = {
            "backend_call_ms": "local_monotonic",
            "decision_finalize_ms": "local_monotonic",
        }
        for stage in ("preprocess_ms", "inference_ms", "postprocess_ms"):
            if backend_stage_timings[stage] is not None:
                provenance[stage] = "backend_reported"
        if capture_to_receive_ms is not None:
            provenance["capture_to_receive_ms"] = "local_monotonic"
        if queue_wait_ms is not None:
            provenance["queue_wait_ms"] = "local_monotonic"
        if end_to_end_ms is not None:
            provenance["end_to_end_ms"] = "local_monotonic"
        timing = VisualTiming(
            capture_to_receive_ms=capture_to_receive_ms,
            queue_wait_ms=queue_wait_ms,
            decode_ms=None,
            preprocess_ms=backend_stage_timings["preprocess_ms"],
            backend_call_ms=(backend_finished_s - backend_started_s) * 1000.0,
            inference_ms=backend_stage_timings["inference_ms"],
            postprocess_ms=backend_stage_timings["postprocess_ms"],
            decision_finalize_ms=(finalized_s - backend_finished_s) * 1000.0,
            end_to_end_ms=end_to_end_ms,
            queue_depth=timing_context.queue_depth,
            deadline_ms=timing_context.deadline_ms,
            deadline_missed=deadline_missed,
            model_id=self.model_id,
            runtime_backend=getattr(self, "runtime_backend", "ultralytics"),
            device=str(self.device),
            input_width=input_width,
            input_height=input_height,
            detection_count=len(detections),
            timing_provenance=provenance,
        )
        return VisualDetectionResult(tuple(detections), timing)

    def export_onnx(self, output_directory=None):
        exported = self.model.export(
            format="onnx",
            imgsz=getattr(self, "image_size", 640),
            dynamic=False,
        )
        path = Path(exported)
        if output_directory is not None:
            destination = Path(output_directory) / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            path = destination
        return path


def _backend_stage_timings(results):
    values = {
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
    }
    if not results:
        return values
    speed = getattr(results[0], "speed", None)
    if not isinstance(speed, dict):
        return values
    mapping = {
        "preprocess_ms": "preprocess",
        "inference_ms": "inference",
        "postprocess_ms": "postprocess",
    }
    for output_name, backend_name in mapping.items():
        value = speed.get(backend_name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
        ):
            values[output_name] = float(value)
    return values


def _input_dimensions(image, frame):
    if frame is not None:
        return frame.width, frame.height
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return None, None
