"""Run visual YOLO validation and operational threshold evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, git_commit, write_json
from src.ml.visual_detection_metrics import (
    select_confidence_threshold,
    threshold_metrics,
)
from src.ml.visual_training_identity import TrainingViewIdentity


PREDICTION_CONFIDENCE_FLOOR = 0.05


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
    model_path,
    dataset_root,
    partition,
    *,
    device,
    imgsz,
    confidence=PREDICTION_CONFIDENCE_FLOOR,
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
        plots=True,
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
    result = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist() if confusion is not None else None,
    }
    if set(per_class) != set(EQUIPMENT_CLASSES):
        raise ValueError("evaluation did not produce metrics for every class")
    _require_finite(result)
    return result


def _require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evaluation metrics contain a non-finite value")


def _formal_commit():
    commit = git_commit()
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("formal visual evaluation requires a clean tracked worktree")
    return commit


def _full_validation_provenance(dataset_root):
    identity = TrainingViewIdentity.from_record(
        json.loads(
            (dataset_root / "identity/training_view_identity.json").read_text()
        )
    )
    return {
        "training_view_identity_sha256": identity.training_view_identity_sha256,
        "source_development_dataset_identity": (
            identity.source_development_dataset_identity
        ),
        "membership_sha256": identity.full_validation_membership_sha256,
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
    output_path = Path(output_path)
    heldout_receipt = None
    if partition == "heldout_test":
        if output_path.exists():
            raise ValueError("held-out evaluation result already exists")
        receipt_path = dataset_root / "identity/heldout_access_receipt.json"
        if not receipt_path.is_file():
            raise ValueError(
                "held-out evaluation requires a frozen-package access receipt"
            )
        heldout_receipt = json.loads(receipt_path.read_text())
        if file_sha256(Path(model_path)) != heldout_receipt.get(
            "canonical_model_sha256"
        ):
            raise ValueError(
                "held-out evaluation requires the canonical frozen model"
            )
    evaluation_commit = _formal_commit()
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
    if partition == "full_validation":
        confidence = select_confidence_threshold(frames)
        threshold_source = "full_validation_macro_f1_sweep"
        dataset_provenance = _full_validation_provenance(dataset_root)
    elif partition == "heldout_test":
        frozen = heldout_receipt.get("frozen_confidence_threshold")
        if not isinstance(frozen, (int, float)):
            raise ValueError("held-out receipt lacks a frozen confidence threshold")
        confidence = {"frozen": threshold_metrics(frames, float(frozen))}
        threshold_source = "frozen_model_package"
        dataset_provenance = {
            "heldout_dataset_identity_sha256": heldout_receipt[
                "heldout_dataset_identity_sha256"
            ],
            "membership_sha256": heldout_receipt["membership_sha256"],
        }
    else:
        confidence = {"fixed": threshold_metrics(frames, 0.25)}
        threshold_source = "fixed_diagnostic_threshold"
        dataset_provenance = _full_validation_provenance(dataset_root)
    result = {
        "visual_evaluation_schema_version": 1,
        "model_sha256": file_sha256(Path(model_path)),
        "partition": partition,
        "frame_count": len(frames),
        "input_size": imgsz,
        "device": device,
        "evaluation_code_commit_sha": evaluation_commit,
        "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "dataset_provenance": dataset_provenance,
        "heldout_access_receipt": heldout_receipt,
        "standard_metrics": standard,
        "confidence_threshold_source": threshold_source,
        "confidence_evaluation": confidence,
    }
    _require_finite(result)
    write_json(output_path, result)
    return result
