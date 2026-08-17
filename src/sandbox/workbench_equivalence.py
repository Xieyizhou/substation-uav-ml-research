"""Bounded PT/ONNX equivalence checks for development workbench runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import tempfile

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256
from src.vision.evaluation.detection_metrics import onnx_equivalence
from src.vision.evaluation.yolo_evaluation import (
    PREDICTION_CONFIDENCE_FLOOR,
    _standard_metrics,
    collect_predictions,
)


def _class_key(label_path):
    rows = Path(label_path).read_text(encoding="utf-8").splitlines()
    if not rows:
        return "no_target"
    class_id = int(rows[0].split()[0])
    return EQUIPMENT_CLASSES[class_id]


def _evenly(rows, limit):
    rows = sorted(rows, key=lambda row: row.name)
    if len(rows) <= limit:
        return rows
    return [
        rows[min(len(rows) - 1, ((2 * index + 1) * len(rows)) // (2 * limit))]
        for index in range(limit)
    ]


def calibration_images(view, total=200):
    groups = defaultdict(list)
    for image in (Path(view) / "images/validation").iterdir():
        if image.is_file():
            label = Path(view) / "labels/validation" / f"{image.stem}.txt"
            groups[_class_key(label)].append(image)
    keys = [*EQUIPMENT_CLASSES, "no_target"]
    quota = total // len(keys)
    selected = []
    for key in keys:
        selected.extend(_evenly(groups[key], min(quota, len(groups[key]))))
    if len(selected) < total:
        used = {path.name for path in selected}
        remainder = [path for rows in groups.values() for path in rows if path.name not in used]
        selected.extend(_evenly(remainder, min(total - len(selected), len(remainder))))
    return sorted(selected[:total], key=lambda path: path.name)


def _calibration_view(view, root, images):
    for image in images:
        label = Path(view) / "labels/validation" / f"{image.stem}.txt"
        destination = root / "images/calibration" / image.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(image)
        except OSError:
            shutil.copy2(image, destination)
        label_destination = root / "labels/calibration" / label.name
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            label_destination.hardlink_to(label)
        except OSError:
            shutil.copy2(label, label_destination)
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES))
    yaml = root / "dataset.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/calibration\nval: images/calibration\nnames:\n{names}",
        encoding="utf-8",
    )
    return yaml


def evaluate_workbench_equivalence(best, onnx, view, *, imgsz, device):
    selected = calibration_images(view)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        yaml = _calibration_view(view, root, selected)
        pt_metrics = _standard_metrics(
            best, yaml, split="val", device=device, imgsz=imgsz,
            output_root=root / "pt_metrics",
        )
        onnx_metrics = _standard_metrics(
            onnx, yaml, split="val", device="cpu", imgsz=imgsz,
            output_root=root / "onnx_metrics",
        )
        pt_frames = collect_predictions(
            best, root, "calibration", device=device, imgsz=imgsz,
            confidence=PREDICTION_CONFIDENCE_FLOOR,
        )
        onnx_frames = collect_predictions(
            onnx, root, "calibration", device="cpu", imgsz=imgsz,
            confidence=PREDICTION_CONFIDENCE_FLOOR,
        )
    detections = onnx_equivalence(pt_frames, onnx_frames)
    map_difference = abs(pt_metrics["mAP50_95"] - onnx_metrics["mAP50_95"])
    return {
        "workbench_onnx_gate_schema_version": 1,
        "input_size": imgsz,
        "calibration_frame_count": len(selected),
        "calibration_membership": [path.stem for path in selected],
        "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "pt_model_sha256": file_sha256(best),
        "onnx_model_sha256": file_sha256(onnx),
        "pt_metrics": pt_metrics,
        "onnx_metrics": onnx_metrics,
        "mAP50_95_absolute_difference": map_difference,
        "detections": detections,
        "passed": detections["passed"] and map_difference <= 0.001,
    }
