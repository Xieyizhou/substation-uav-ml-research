"""Run visual YOLO validation and operational threshold evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, write_json
from src.ml.visual_detection_metrics import select_confidence_threshold


def _truth(label_path, width=1920, height=1080, input_size=640):
    rows = []
    scale = min(input_size / width, input_size / height)
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, box_width, box_height = map(float, line.split())
        box_width *= width
        box_height *= height
        cx *= width
        cy *= height
        rows.append(
            {
                "class_name": EQUIPMENT_CLASSES[int(class_id)],
                "bbox": [
                    cx - box_width / 2,
                    cy - box_height / 2,
                    cx + box_width / 2,
                    cy + box_height / 2,
                ],
                "small": box_width * box_height * scale * scale < 32 * 32,
            }
        )
    return rows


def _prediction_rows(result):
    if result.boxes is None:
        return []
    rows = []
    for index, class_id in enumerate(result.boxes.cls.tolist()):
        rows.append(
            {
                "class_name": EQUIPMENT_CLASSES[int(class_id)],
                "confidence": float(result.boxes.conf[index]),
                "bbox": [float(value) for value in result.boxes.xyxy[index].tolist()],
            }
        )
    return rows


def collect_predictions(
    model_path, dataset_root, partition, *, device, imgsz, confidence=0.05
):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("visual evaluation requires requirements-ml.txt") from error
    dataset_root = Path(dataset_root)
    image_root = dataset_root / "images" / partition
    label_root = dataset_root / "labels" / partition
    model = YOLO(str(model_path))
    frames = []
    for result in model.predict(
        source=str(image_root),
        stream=True,
        imgsz=imgsz,
        conf=confidence,
        iou=0.7,
        device=device,
        verbose=False,
    ):
        stem = Path(result.path).stem
        frames.append(
            {
                "sample_id": stem,
                "truth": _truth(label_root / f"{stem}.txt", input_size=imgsz),
                "predictions": _prediction_rows(result),
            }
        )
    return frames


def _standard_metrics(model_path, dataset_yaml, *, split, device, imgsz):
    from ultralytics import YOLO

    metrics = YOLO(str(model_path)).val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        device=device,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    names = metrics.names
    per_class = {}
    for index, class_id in enumerate(getattr(box, "ap_class_index", [])):
        per_class[names[int(class_id)]] = {
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
            "mAP50": float(box.ap50[index]),
            "mAP50_95": float(box.ap[index]),
        }
    confusion = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist() if confusion is not None else None,
    }


def evaluate_yolo(
    model_path,
    dataset_root,
    output_path,
    *,
    partition="full_validation",
    device="mps",
    imgsz=640,
):
    dataset_root = Path(dataset_root)
    heldout_receipt = None
    if partition == "heldout_test":
        receipt_path = dataset_root / "identity/heldout_access_receipt.json"
        if not receipt_path.is_file():
            raise ValueError(
                "held-out evaluation requires a frozen-package access receipt"
            )
        heldout_receipt = json.loads(receipt_path.read_text())
        if file_sha256(Path(model_path)) not in heldout_receipt.get(
            "allowed_model_sha256", []
        ):
            raise ValueError(
                "held-out model is not part of the package that unlocked the view"
            )
    split = (
        "test"
        if partition in {"full_validation", "heldout_test"}
        else "val"
    )
    standard = _standard_metrics(
        model_path,
        dataset_root / "dataset.yaml",
        split=split,
        device=device,
        imgsz=imgsz,
    )
    frames = collect_predictions(
        model_path, dataset_root, partition, device=device, imgsz=imgsz
    )
    threshold = select_confidence_threshold(frames)
    result = {
        "visual_evaluation_schema_version": 1,
        "model_sha256": file_sha256(Path(model_path)),
        "partition": partition,
        "frame_count": len(frames),
        "input_size": imgsz,
        "device": device,
        "heldout_access_receipt": heldout_receipt,
        "standard_metrics": standard,
        "confidence_selection": threshold,
    }
    write_json(Path(output_path), result)
    return result
