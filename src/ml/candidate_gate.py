"""Readiness audit and bounded validation replay for LiDAR candidates."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

from src.ml import RISK_LABELS
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.dataset import load_dataset
from src.ml.dataset_builder import validate_dataset_directory
from src.ml.model_package import validate_model_package
from src.ml.predictions import evaluate_predictions, predict_dataset
from src.study.comparison import replay_gate


GATE_VERSION = "lidar-validation-replay-gate-v1"


def _check(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def _split_counts(samples):
    return {
        split: dict(Counter(sample.risk_label for sample in samples if sample.split == split))
        for split in ("train", "validation", "test")
    }


def audit_candidate(package_dir, dataset_dir):
    package_dir, dataset_dir = Path(package_dir), Path(dataset_dir)
    manifest = validate_model_package(package_dir)
    dataset = validate_dataset_directory(dataset_dir)
    samples = load_dataset(dataset_dir / dataset["samples_file"])
    counts = _split_counts(samples)
    training = manifest.get("training_parameters", {})
    metrics = manifest.get("offline_metrics", {})
    validation = metrics.get("validation", {})
    onnx = metrics.get("onnx", {})
    coverage = {
        split: all(counts[split].get(label, 0) > 0 for label in RISK_LABELS)
        for split in counts
    }
    per_class = validation.get("per_class", {})
    class_metrics = all(
        per_class.get(label, {}).get("support", 0) > 0
        and math.isfinite(float(per_class.get(label, {}).get("recall", float("nan"))))
        for label in RISK_LABELS
    )
    latency = onnx.get("cpu_latency", {}).get("p95_ms")
    checks = [
        _check(
            "dataset_identity",
            manifest.get("dataset_id") == dataset.get("dataset_id")
            and manifest.get("dataset_sha256") == dataset.get("data_sha256"),
            "model package must bind the current dataset identity and hash",
        ),
        _check(
            "clean_training_commit",
            bool(manifest.get("training_commit"))
            and not str(manifest.get("training_commit")).endswith("-dirty"),
            str(manifest.get("training_commit")),
        ),
        _check(
            "complete_training_labels",
            not training.get("allow_incomplete_labels", False),
            "allow_incomplete_labels must be false",
        ),
        _check("split_label_coverage", all(coverage.values()), counts),
        _check("validation_class_metrics", class_metrics, per_class),
        _check(
            "onnx_equivalence",
            onnx.get("max_absolute_error") is not None
            and float(onnx["max_absolute_error"]) <= 1e-4,
            onnx.get("max_absolute_error"),
        ),
        _check(
            "onnx_latency",
            latency is not None and math.isfinite(float(latency)) and float(latency) <= 50.0,
            latency,
        ),
        _check(
            "onnx_external_data",
            not (package_dir / "model.onnx.data").exists()
            or bool(manifest.get("onnx_external_data_sha256")),
            manifest.get("onnx_external_data_sha256"),
        ),
    ]
    return {
        "gate_version": GATE_VERSION,
        "passed": all(item["passed"] for item in checks),
        "model_id": manifest["model_id"],
        "dataset_id": dataset["dataset_id"],
        "split_label_counts": counts,
        "checks": checks,
    }


def run_validation_replay(package_dir, dataset_dir, output_dir):
    """Run the fixed validation split only; held-out data is not accessed."""
    package_dir, dataset_dir, output_dir = map(Path, (package_dir, dataset_dir, output_dir))
    readiness = audit_candidate(package_dir, dataset_dir)
    if not readiness["passed"]:
        failed = [item["name"] for item in readiness["checks"] if not item["passed"]]
        raise ValueError("candidate readiness failed: " + ", ".join(failed))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / "predictions.validation.jsonl"
    predict_dataset(
        package_dir / "model.onnx",
        dataset_dir / "samples.jsonl",
        predictions,
        split="validation",
    )
    metrics = evaluate_predictions(predictions)
    candidate = {
        "risk_f1": metrics["macro_f1"],
        "danger_recall": metrics["per_class"]["danger"]["recall"],
        "traversability_iou": metrics["traversability_iou"],
        "risk_ece": metrics["ece"],
        "inference_p95_ms": metrics["latency"]["p95_ms"],
    }
    decision = replay_gate(candidate)
    if candidate["risk_f1"] < 0.5:
        decision["reasons"].append("validation macro F1 is below 0.50")
    if candidate["danger_recall"] < 0.6:
        decision["reasons"].append("validation danger recall is below 0.60")
    decision["passed"] = not decision["reasons"]
    record = {
        "gate_version": GATE_VERSION,
        "partition": "validation",
        "passed": decision["passed"],
        "reasons": decision["reasons"],
        "model_id": readiness["model_id"],
        "dataset_id": readiness["dataset_id"],
        "model_sha256": file_sha256(package_dir / "model.onnx"),
        "dataset_sha256": file_sha256(dataset_dir / "samples.jsonl"),
        "predictions_sha256": file_sha256(predictions),
        "metrics": metrics,
        "gate_metrics": candidate,
    }
    record["replay_gate_identity_sha256"] = object_sha256(record)
    write_json(output_dir / "replay_gate.json", record)
    return record
