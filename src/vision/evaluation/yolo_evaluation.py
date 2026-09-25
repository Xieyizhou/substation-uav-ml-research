"""Run visual YOLO validation and operational threshold evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, git_commit, write_json
from src.vision.evaluation.detection_metrics import (
    select_confidence_threshold,
    threshold_metrics,
)
from src.vision.contracts.training_identity import TrainingViewIdentity


from src.vision.evaluation.yolo_predictions import (
    PREDICTION_CONFIDENCE_FLOOR, collect_predictions,
    _truth, _prediction_rows, _timed_results,
)
RUNTIME_PREDICTION_BATCH_SIZE = 1
RUNTIME_DIAGNOSTIC_THRESHOLD = 0.25


def _standard_metrics(
    model_path, dataset_yaml, *, split, device, imgsz, output_root=None,
    required_classes=None,
):
    from ultralytics import YOLO

    output_options = {}
    if output_root is not None:
        output_root = Path(output_root)
        output_options = {
            "project": str(output_root.parent), "name": output_root.name,
            "exist_ok": True,
        }
    metrics = YOLO(str(model_path)).val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        device=device,
        plots=True,
        rect=False,
        verbose=False,
        **output_options,
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
    expected_classes = set(
        EQUIPMENT_CLASSES if required_classes is None else required_classes
    )
    if set(per_class) != expected_classes:
        raise ValueError("evaluation did not produce metrics for every class")
    _require_finite(result)
    return result


def _postprocessing_consistency(standard, runtime_fixed):
    """Reject validator AP that masks an unreachable runtime class.

    Ultralytics validation uses multi-label NMS while the predictor used by the
    product is single-label. Validator AP remains diagnostic, but a class with
    excellent diagnostic AP and unusable runtime recall is an explicit
    contradiction that blocks promotion.
    """
    per_class = {}
    diagnostic = standard.get("per_class", {})
    for name in EQUIPMENT_CLASSES:
        diagnostic_map50 = float(diagnostic.get(name, {}).get("mAP50", 0.0))
        runtime_recall = float(runtime_fixed["per_class"][name]["recall"])
        contradiction = diagnostic_map50 >= 0.90 and runtime_recall < 0.50
        per_class[name] = {
            "diagnostic_multi_label_map50": diagnostic_map50,
            "runtime_single_label_recall": runtime_recall,
            "contradiction": contradiction,
            "passed": not contradiction,
        }
    return {
        "threshold": RUNTIME_DIAGNOSTIC_THRESHOLD,
        "per_class": per_class,
        "passed": all(row["passed"] for row in per_class.values()),
    }


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
    prediction_batch = RUNTIME_PREDICTION_BATCH_SIZE
    frames = collect_predictions(
        model_path, dataset_root, partition, device=device, imgsz=imgsz,
        batch=prediction_batch,
    )
    runtime_fixed = threshold_metrics(frames, RUNTIME_DIAGNOSTIC_THRESHOLD)
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
        "visual_evaluation_schema_version": 2,
        "model_sha256": file_sha256(Path(model_path)),
        "partition": partition,
        "frame_count": len(frames),
        "input_size": imgsz,
        "device": device,
        "evaluation_code_commit_sha": evaluation_commit,
        "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "prediction_batch_size": prediction_batch,
        "dataset_provenance": dataset_provenance,
        "heldout_access_receipt": heldout_receipt,
        "standard_metrics": standard,
        "standard_metrics_role": "diagnostic_multi_label_validator_only",
        "runtime_postprocessing": {
            "nms": "single_label_predictor",
            "batch_size": RUNTIME_PREDICTION_BATCH_SIZE,
            "promotion_authority": True,
        },
        "runtime_diagnostic_metrics": runtime_fixed,
        "postprocessing_consistency": _postprocessing_consistency(
            standard, runtime_fixed
        ),
        "confidence_threshold_source": threshold_source,
        "confidence_evaluation": confidence,
    }
    _require_finite(result)
    write_json(output_path, result)
    return result
