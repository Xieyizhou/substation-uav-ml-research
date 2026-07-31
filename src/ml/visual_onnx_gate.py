"""PT/ONNX equivalence gate on a fixed validation calibration view."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import write_json
from src.ml.visual_detection_metrics import onnx_equivalence
from src.ml.visual_training_view import evenly_select
from src.ml.visual_yolo_dataset import link_image
from src.ml.visual_yolo_evaluation import (
    _standard_metrics,
    collect_predictions,
)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def calibration_members(rows, total=200):
    groups = defaultdict(list)
    for row in rows:
        key = row["classes"][0] if row["classes"] else "no_target"
        groups[key].append(row)
    keys = [*EQUIPMENT_CLASSES, "no_target"]
    per_group = total // len(keys)
    selected = []
    for key in keys:
        selected.extend(evenly_select(groups[key], min(per_group, len(groups[key]))))
    if len(selected) < total:
        used = {row["sample_id"] for row in selected}
        remainder = [row for row in rows if row["sample_id"] not in used]
        selected.extend(evenly_select(remainder, min(total - len(selected), len(remainder))))
    return sorted(selected[:total], key=lambda row: row["sample_id"])


def _calibration_dataset(dataset_root, root, rows):
    for row in rows:
        image = Path(dataset_root) / row["image_relative_path"]
        label = Path(dataset_root) / row["label_relative_path"]
        link_image(image, root / "images/calibration" / image.name)
        link_image(label, root / "labels/calibration" / label.name)
    yaml = root / "dataset.yaml"
    yaml.write_text(
        f"path: {root}\nval: images/calibration\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES)),
        encoding="utf-8",
    )
    return yaml


def validate_onnx_equivalence(
    pt_model,
    onnx_model,
    dataset_root,
    output_path,
    *,
    imgsz=640,
    device="cpu",
):
    rows = _read_jsonl(Path(dataset_root) / "identity/validation_membership.jsonl")
    selected = calibration_members(rows)
    with tempfile.TemporaryDirectory() as directory:
        calibration_root = Path(directory)
        yaml = _calibration_dataset(dataset_root, calibration_root, selected)
        pt_metrics = _standard_metrics(
            pt_model, yaml, split="val", device=device, imgsz=imgsz
        )
        onnx_metrics = _standard_metrics(
            onnx_model, yaml, split="val", device="cpu", imgsz=imgsz
        )
        pt_frames = collect_predictions(
            pt_model,
            calibration_root,
            "calibration",
            device=device,
            imgsz=imgsz,
            confidence=0.25,
        )
        onnx_frames = collect_predictions(
            onnx_model,
            calibration_root,
            "calibration",
            device="cpu",
            imgsz=imgsz,
            confidence=0.25,
        )
    equivalence = onnx_equivalence(pt_frames, onnx_frames)
    map_difference = abs(
        pt_metrics["mAP50_95"] - onnx_metrics["mAP50_95"]
    )
    result = {
        "onnx_equivalence_schema_version": 1,
        "input_size": imgsz,
        "calibration_frame_count": len(selected),
        "pt_metrics": pt_metrics,
        "onnx_metrics": onnx_metrics,
        "mAP50_95_absolute_difference": map_difference,
        "detections": equivalence,
        "passed": equivalence["passed"] and map_difference <= 0.001,
    }
    write_json(Path(output_path), result)
    if not result["passed"]:
        raise ValueError("PT/ONNX equivalence gate failed")
    return result
