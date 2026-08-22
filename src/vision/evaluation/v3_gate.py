"""Real-validation and synthetic-regression gates for YOLO11n v3."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.evaluation.detection_metrics import threshold_metrics
from src.vision.evaluation.yolo_evaluation import (
    _standard_metrics,
    collect_predictions,
)


def _selected(record):
    selected = record.get("confidence_evaluation", {}).get("selected")
    if not isinstance(selected, dict):
        raise ValueError("real validation result lacks a frozen threshold")
    return selected


def evaluate_synthetic_regression(model_path, dataset_root,
                                  real_validation_results, output_path,
                                  *, device="mps", imgsz=640):
    validation = json.loads(Path(real_validation_results).read_text())
    if validation.get("partition") != "full_validation":
        raise ValueError("v3 threshold must come from full real validation")
    if validation.get("model_sha256") != file_sha256(Path(model_path)):
        raise ValueError("real validation references different model weights")
    threshold = float(_selected(validation)["threshold"])
    dataset_root = Path(dataset_root)
    frames = collect_predictions(
        model_path, dataset_root, "synthetic_regression",
        device=device, imgsz=imgsz, batch=16,
    )
    result = {
        "v3_synthetic_regression_schema_version": 1,
        "model_sha256": file_sha256(Path(model_path)),
        "training_view_identity_sha256": json.loads(
            (dataset_root / "identity/training_view_identity.json").read_text()
        )["training_view_identity_sha256"],
        "membership_sha256": file_sha256(
            dataset_root / "identity/synthetic_regression_membership.jsonl"
        ),
        "frame_count": len(frames), "input_size": imgsz,
        "frozen_real_validation_threshold": threshold,
        "standard_metrics": _standard_metrics(
            model_path, dataset_root / "dataset-synthetic-regression.yaml",
            split="val", device=device, imgsz=imgsz,
        ),
        "threshold_metrics": threshold_metrics(frames, threshold),
    }
    write_json(output_path, result)
    return result


def validate_v3_candidate(real_validation_results, synthetic_regression_results,
                          v2_synthetic_baseline, output_path):
    real = json.loads(Path(real_validation_results).read_text())
    regression = json.loads(Path(synthetic_regression_results).read_text())
    baseline = json.loads(Path(v2_synthetic_baseline).read_text())
    if real.get("model_sha256") != regression.get("model_sha256"):
        raise ValueError("real and synthetic results reference different weights")
    selected = _selected(real)
    standard = real.get("standard_metrics", {})
    checks = {
        "real_map50_95": float(standard.get("mAP50_95", math.nan)) >= 0.45,
        "real_macro_f1": float(selected.get("macro_f1", math.nan)) >= 0.65,
        "real_per_class_recall": all(
            float(selected.get("per_class", {}).get(name, {}).get("recall", math.nan)) >= 0.50
            for name in EQUIPMENT_CLASSES
        ),
        "real_small_object_recall": (
            isinstance(selected.get("small_object_recall"), (int, float))
            and selected["small_object_recall"] >= 0.35
        ),
        "real_no_target_fpr": (
            isinstance(selected.get("no_target_false_positive_rate"), (int, float))
            and selected["no_target_false_positive_rate"] <= 0.10
        ),
    }
    regression_threshold = regression.get("threshold_metrics", {})
    baseline_selected = _selected(baseline)
    checks["synthetic_macro_f1_regression"] = (
        float(regression_threshold.get("macro_f1", math.nan))
        >= float(baseline_selected.get("macro_f1", math.nan)) - 0.10
    )
    checks["synthetic_no_target_fpr"] = (
        isinstance(regression_threshold.get("no_target_false_positive_rate"), (int, float))
        and regression_threshold["no_target_false_positive_rate"] <= 0.05
    )
    record = {
        "real_domain_v3_gate_schema_version": 1,
        "model_sha256": real["model_sha256"],
        "training_view_identity_sha256": regression["training_view_identity_sha256"],
        "frozen_confidence_threshold": selected["threshold"],
        "real_validation_results_sha256": file_sha256(Path(real_validation_results)),
        "synthetic_regression_results_sha256": file_sha256(
            Path(synthetic_regression_results)
        ),
        "v2_synthetic_baseline_sha256": file_sha256(Path(v2_synthetic_baseline)),
        "checks": checks,
        "passed": all(checks.values()),
    }
    record["gate_identity_sha256"] = object_sha256(record)
    write_json(output_path, record)
    return record
