"""Quality metrics derived from identity-bound LiDAR flight logs."""

from __future__ import annotations

import json
import math

from src.ml import RISK_LABELS
from src.ml.metrics import (
    binary_iou,
    classification_report,
    expected_calibration_error,
)


def _risk(value):
    value = str(value or "").strip().lower()
    return "clear" if value == "detected" else value


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_vector(value):
    try:
        result = json.loads(str(value))
        return [float(item) for item in result]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def lidar_quality_metrics(frame):
    """Return formal risk, calibration, traversability, and direction metrics."""
    if "truth_risk_level" not in frame or "perception_risk_level" not in frame:
        return {}
    rows = []
    for _, row in frame.iterrows():
        truth, prediction = _risk(row.get("truth_risk_level")), _risk(
            row.get("perception_risk_level")
        )
        if truth in RISK_LABELS and prediction in RISK_LABELS:
            rows.append((row, truth, prediction))
    if not rows:
        return {}
    labels = [truth for _, truth, _ in rows]
    predictions = [prediction for _, _, prediction in rows]
    report = classification_report(labels, predictions, RISK_LABELS)
    confidence_rows = [
        (truth, prediction, _finite(row.get("risk_confidence")))
        for row, truth, prediction in rows
    ]
    confidence_rows = [item for item in confidence_rows if item[2] is not None]
    ece = (
        expected_calibration_error(
            [item[0] for item in confidence_rows],
            [item[2] for item in confidence_rows],
            [item[1] for item in confidence_rows],
        )
        if confidence_rows
        else None
    )
    safety = [item for item in rows if item[1] in {"warning", "danger"}]
    false_negatives = sum(item[2] == "clear" for item in safety)
    traversability, direction = [], []
    for row, _, _ in rows:
        truth_map = _json_vector(row.get("truth_traversability_json"))
        prediction_map = _json_vector(row.get("predicted_traversability_json"))
        if truth_map and prediction_map and len(truth_map) == len(prediction_map):
            traversability.append(binary_iou(truth_map, prediction_map))
        truth_direction = _finite(row.get("truth_direction_deg"))
        predicted_direction = _finite(row.get("predicted_direction_deg"))
        if truth_direction is not None and predicted_direction is not None:
            direction.append(abs(truth_direction - predicted_direction))
    return {
        "risk_f1": report["macro_f1"],
        "danger_recall": report["per_class"]["danger"]["recall"],
        "risk_false_negative_rate": false_negatives / max(len(safety), 1),
        "risk_ece": ece,
        "traversability_iou": (
            sum(traversability) / len(traversability) if traversability else None
        ),
        "direction_mae_deg": sum(direction) / len(direction) if direction else None,
        "quality_sample_count": len(rows),
    }
