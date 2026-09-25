"""Compare on the same validation images, preferring the actual parent weights."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.evaluation.detection_metrics import threshold_metrics, select_confidence_threshold
from src.vision.evaluation.yolo_evaluation import collect_predictions


def _delta(candidate, baseline, key):
    left, right = candidate.get(key), baseline.get(key)
    return None if left is None or right is None else left - right


def _baseline(project_root, run_root, recipe):
    initialization = recipe.initialization or {}
    if initialization:
        from src.sandbox.workbench_weights import resolve_initial_weights
        model = resolve_initial_weights(project_root, recipe)
        parent_id = initialization.get("parent_experiment_id")
        threshold = initialization.get("parent_threshold")
        if parent_id and threshold is None:
            # Old recipes remain unchanged; verify their parent before reading
            # its threshold. Never silently compare them to an unrelated v1.
            from src.sandbox.workbench_inference import verified_model
            if Path(parent_id).name != parent_id:
                raise ValueError("invalid comparison parent identifier")
            parent = verified_model(Path(run_root).parent / parent_id)
            if (parent["receipt_identity_sha256"] != initialization["parent_receipt_identity_sha256"]
                    or parent["best_weights_sha256"] != recipe.pretrained_weights_sha256):
                raise ValueError("comparison parent identity changed")
            threshold = parent["threshold"]
        return model, threshold, dict(
            baseline_kind="parent_model" if parent_id else "initial_weights",
            baseline_experiment_id=parent_id,
            baseline_receipt_identity_sha256=initialization.get("parent_receipt_identity_sha256"),
            baseline_format="pytorch", threshold_policy="frozen_parent" if parent_id else "selection_validation",
        )
    package = Path(project_root) / "models/equipment/visual-yolo11n-baseline-v1-package"
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        return None, None, {"status": "baseline_unavailable"}
    from src.vision.evaluation.yolo_package import validate_yolo_package

    manifest = validate_yolo_package(package)
    expected = recipe.baseline_package_identity_sha256
    if expected and manifest["package_identity_sha256"] != expected:
        raise ValueError("comparison baseline package identity changed")
    size = str(recipe.parameters["imgsz"])
    export = manifest.get("exports", {}).get(size)
    if not export:
        return None, None, {"status": "baseline_size_unavailable", "input_size": int(size)}
    return package / export["path"], float(manifest["frozen_confidence_threshold"]), dict(
        baseline_kind="legacy_package", baseline_format="onnx", threshold_policy="frozen_package",
        baseline_package_identity_sha256=manifest["package_identity_sha256"],
    )


def compare_with_baseline(project_root, view, run_root, recipe, candidate_replay):
    model, baseline_threshold, source = _baseline(project_root, run_root, recipe)
    if model is None:
        value = {"workbench_comparison_schema_version": 1, **source}
        write_json(Path(run_root) / "comparison.json", value)
        return value
    size = recipe.parameters["imgsz"]
    frames = collect_predictions(
        model, view, "validation", device="cpu", imgsz=int(size), batch=1,
    )
    if baseline_threshold is None:
        baseline_threshold = select_confidence_threshold(frames)["selected"]["threshold"]
    if not 0 <= baseline_threshold <= 1:
        raise ValueError("invalid comparison confidence threshold")
    if len(frames) != candidate_replay["frame_count"]:
        raise ValueError("comparison validation frame count differs")
    baseline = threshold_metrics(frames, baseline_threshold)
    candidate = candidate_replay["metrics"]
    value = {
        "workbench_comparison_schema_version": 1,
        "status": "complete", "input_size": int(size),
        "membership": "candidate_validation_view",
        "membership_sha256": file_sha256(Path(view) / "membership.json"),
        "frame_count": len(frames), "candidate_format": "onnx",
        "scope": "development validation; candidate threshold selected here; not held-out or flight qualification",
        **source,
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
