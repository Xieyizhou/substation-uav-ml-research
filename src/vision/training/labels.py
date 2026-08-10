"""Convert simulator bounding-box records to a versioned YOLO dataset."""

from __future__ import annotations

from dataclasses import dataclass

from src.ml import EQUIPMENT_CLASSES


@dataclass(frozen=True)
class BoundingBoxLabel:
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_width: int
    image_height: int

    def to_yolo(self):
        if self.class_name not in EQUIPMENT_CLASSES:
            raise ValueError(f"unsupported equipment class: {self.class_name}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        x_min = min(max(self.x_min, 0.0), self.image_width)
        y_min = min(max(self.y_min, 0.0), self.image_height)
        x_max = min(max(self.x_max, 0.0), self.image_width)
        y_max = min(max(self.y_max, 0.0), self.image_height)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("bounding box has no visible area")
        center_x = ((x_min + x_max) / 2.0) / self.image_width
        center_y = ((y_min + y_max) / 2.0) / self.image_height
        width = (x_max - x_min) / self.image_width
        height = (y_max - y_min) / self.image_height
        class_id = EQUIPMENT_CLASSES.index(self.class_name)
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
