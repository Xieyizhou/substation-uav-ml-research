"""Four-class YOLO equipment detector with optional ONNX export."""

from __future__ import annotations

from pathlib import Path
import time

from src.ml import EQUIPMENT_CLASSES
from src.sensors.types import EquipmentDetection


class EquipmentDetector:
    def __init__(self, weights, confidence=0.25, device="cpu"):
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

    def detect(self, image, *, timestamp_s=None, frame_id="research_rgb"):
        timestamp_s = time.time() if timestamp_s is None else timestamp_s
        results = self.model.predict(
            source=image,
            imgsz=640,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
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
        return tuple(detections)

    def export_onnx(self, output_directory=None):
        exported = self.model.export(format="onnx", imgsz=640, dynamic=False)
        path = Path(exported)
        if output_directory is not None:
            destination = Path(output_directory) / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            path = destination
        return path
