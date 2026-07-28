"""ONNX dataset prediction and dependency-free offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import time

from src.ml import RISK_LABELS
from src.ml.dataset import load_dataset
from src.ml.metrics import (
    binary_iou,
    classification_report,
    expected_calibration_error,
    latency_summary,
)
from src.ml.onnx_risk import OnnxRiskModel
from src.sensors.types import LaserScanFrame


def read_predictions(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("prediction file is empty")
    return rows


def predict_dataset(model_path, dataset_path, output_path, *, split="test"):
    model = OnnxRiskModel(model_path)
    samples = [sample for sample in load_dataset(dataset_path) if sample.split == split]
    if not samples:
        raise ValueError(f"dataset has no {split} samples")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for index, sample in enumerate(samples, start=1):
            scan = LaserScanFrame(
                timestamp_s=sample.timestamp_s,
                received_monotonic_s=time.monotonic(),
                frame_id="dataset",
                angle_min_rad=-3.141592653589793,
                angle_max_rad=3.141592653589793,
                angle_step_rad=6.283185307179586
                / max(len(sample.ranges_m) - 1, 1),
                range_min_m=0.0,
                range_max_m=sample.range_max_m,
                ranges_m=sample.ranges_m,
                source="dataset",
                sequence=index,
            )
            prediction = model.predict(scan)
            output.write(
                json.dumps(
                    {
                        "scenario_id": sample.scenario_id,
                        "split": sample.split,
                        "risk_label": sample.risk_label,
                        "predicted_risk": prediction["risk_level"],
                        "confidence": prediction["confidence"],
                        "traversability": list(sample.traversability),
                        "predicted_traversability": prediction["traversability"],
                        "recommended_direction_deg": sample.recommended_direction_deg,
                        "predicted_direction_deg": prediction["recommended_direction_deg"],
                        "latency_ms": prediction["latency_ms"],
                        "model_id": prediction["model_id"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return output_path


def evaluate_predictions(path):
    rows = read_predictions(path)
    labels = [row["risk_label"] for row in rows]
    predictions = [row["predicted_risk"] for row in rows]
    confidences = [float(row.get("confidence", 0.0)) for row in rows]
    report = classification_report(labels, predictions, RISK_LABELS)
    report["ece"] = expected_calibration_error(labels, confidences, predictions)
    traversability = [
        binary_iou(row["traversability"], row["predicted_traversability"])
        for row in rows
        if "traversability" in row and "predicted_traversability" in row
    ]
    report["traversability_iou"] = (
        sum(traversability) / len(traversability) if traversability else None
    )
    report["latency"] = latency_summary(
        [float(row["latency_ms"]) for row in rows if "latency_ms" in row]
    )
    return report
