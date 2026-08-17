"""Compare a workbench candidate with the compatible local visual baseline."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.evaluation.detection_metrics import threshold_metrics
from src.vision.evaluation.yolo_evaluation import collect_predictions


def _delta(candidate, baseline, key):
    left, right = candidate.get(key), baseline.get(key)
    return None if left is None or right is None else left - right


def compare_with_baseline(project_root, view, run_root, recipe, candidate_replay):
    package = Path(project_root) / "models/equipment/visual-yolo11n-baseline-v1-package"
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        value = {"workbench_comparison_schema_version": 1,
                 "status": "baseline_unavailable"}
        write_json(Path(run_root) / "comparison.json", value)
        return value
    from src.vision.evaluation.yolo_package import validate_yolo_package

    manifest = validate_yolo_package(package)
    size = str(recipe.parameters["imgsz"])
    export = manifest.get("exports", {}).get(size)
    if not export:
        value = {"workbench_comparison_schema_version": 1,
                 "status": "baseline_size_unavailable", "input_size": int(size)}
        write_json(Path(run_root) / "comparison.json", value)
        return value
    model = package / export["path"]
    frames = collect_predictions(
        model, view, "validation", device="cpu", imgsz=int(size), batch=1,
    )
    baseline_threshold = float(manifest["frozen_confidence_threshold"])
    baseline = threshold_metrics(frames, baseline_threshold)
    candidate = candidate_replay["metrics"]
    value = {
        "workbench_comparison_schema_version": 1,
        "status": "complete", "input_size": int(size),
        "membership": "candidate_validation_view",
        "baseline_model_sha256": file_sha256(model),
        "baseline_threshold": baseline_threshold,
        "candidate_threshold": candidate_replay["threshold"],
        "baseline_metrics": baseline, "candidate_metrics": candidate,
        "deltas": {
            key: _delta(candidate, baseline, key)
            for key in ("macro_f1", "small_object_recall",
                        "no_target_false_positive_rate")
        },
    }
    value["comparison_identity_sha256"] = object_sha256(value)
    write_json(Path(run_root) / "comparison.json", value)
    return value
