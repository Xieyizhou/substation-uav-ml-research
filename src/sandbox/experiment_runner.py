"""Execute non-blind visual recipes and materialize aggregate reports."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import time

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.vision.collection.pilot import _read_jsonl, _write_jsonl
from src.vision.contracts.identity import ModelIdentity
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.evaluation.detection_metrics import threshold_metrics
from src.vision.evaluation.yolo_evaluation import _prediction_rows, _truth
from src.vision.evaluation.yolo_package import validate_yolo_package
from src.vision.replay.static_runtime import runtime_environment, timing_summary
from src.vision.replay.static_source import ordered_replay_sources, write_source_list

from src.sandbox.experiment_recipe import ExperimentRecipe, SELECTION_ALGORITHM


TIMING_KEYS = ("preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms")
WARMUP_FRAME_COUNT = 8


def select_source_rows(rows, limit, algorithm):
    if algorithm != SELECTION_ALGORITHM:
        raise ValueError("unsupported sandbox frame-selection algorithm")
    if limit <= 0 or limit > len(rows):
        raise ValueError("sandbox frame limit exceeds available membership")
    if limit == len(rows):
        return list(rows)
    return [rows[((2 * index + 1) * len(rows)) // (2 * limit)] for index in range(limit)]


def _clean_commit(project_root):
    commit = git_commit(project_root)
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("sandbox experiment execution requires a clean tracked commit")
    return commit


def _approved_recipe_path(project_root, recipe_path):
    root = (Path(project_root) / "outputs/sandbox/experiments").resolve()
    path = Path(recipe_path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("recipe is outside the sandbox experiment root") from error
    if len(relative.parts) != 2 or relative.name != "recipe.json":
        raise ValueError("recipe path must identify one sandbox experiment")
    return path


def _load_context(project_root, recipe_path):
    path = _approved_recipe_path(project_root, recipe_path)
    recipe = ExperimentRecipe.from_record(json.loads(path.read_text()))
    if path.parent.name != recipe.experiment_id:
        raise ValueError("recipe directory and experiment identifier differ")
    if recipe.software_commit_sha != _clean_commit(project_root):
        raise ValueError("recipe references a different software commit")
    dataset_root = Path(project_root) / recipe.dataset_root
    package_root = Path(project_root) / recipe.package_root
    package = validate_yolo_package(package_root)
    if package["package_identity_sha256"] != recipe.package_identity_sha256:
        raise ValueError("recipe references a different model package")
    identity = TrainingViewIdentity.from_record(
        json.loads((dataset_root / "identity/training_view_identity.json").read_text())
    )
    if identity.training_view_identity_sha256 != recipe.training_view_identity_sha256:
        raise ValueError("recipe references a different training view")
    membership = dataset_root / "identity" / f"{recipe.partition}_membership.jsonl"
    if file_sha256(membership) != recipe.membership_sha256:
        raise ValueError("recipe membership hash mismatch")
    rows = select_source_rows(
        _read_jsonl(membership), recipe.frame_limit, recipe.selection_algorithm
    )
    if len(rows) != recipe.source_frame_count:
        raise ValueError("recipe source frame count is unavailable")
    models = json.loads((package_root / "model_identities.json").read_text())
    model = ModelIdentity.from_record(models[str(recipe.input_size)])
    export = package["exports"][str(recipe.input_size)]
    checks = (
        (model.model_identity_sha256, recipe.model_identity_sha256),
        (model.preprocessing_configuration_id, recipe.preprocessing_configuration_id),
        (model.runtime_backend, recipe.runtime_backend),
        (export["sha256"], recipe.model_file_sha256),
        (file_sha256(package_root / export["path"]), recipe.model_file_sha256),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("recipe model or preprocessing identity mismatch")
    return recipe, path.parent, dataset_root, package_root / export["path"], rows


def _execute_model(model_path, rows, paths, source_root, recipe):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("sandbox visual evaluation requires requirements-ml.txt") from error
    selected_rows = rows[:: recipe.frame_skip_interval]
    selected_paths = paths[:: recipe.frame_skip_interval]
    model = YOLO(str(model_path))
    options = {
        "stream": True, "batch": 1, "imgsz": recipe.input_size,
        "conf": recipe.confidence_threshold, "iou": 0.7,
        "device": recipe.device, "rect": False, "verbose": False,
    }
    warmup = selected_paths[: min(WARMUP_FRAME_COUNT, len(selected_paths))]
    if warmup:
        list(model.predict(str(write_source_list(source_root, "warmup", warmup)), **options))
    source = write_source_list(source_root, recipe.experiment_id, selected_paths)
    timings = {name: [] for name in TIMING_KEYS}
    raw, predictions = [], {}
    started = time.perf_counter()
    for row, result in zip(selected_rows, model.predict(str(source), **options), strict=True):
        prediction = _prediction_rows(result)
        speed = result.speed
        preprocess = float(speed.get("preprocess", 0.0))
        inference = float(speed.get("inference", 0.0))
        postprocess = float(speed.get("postprocess", 0.0))
        timings["preprocess_ms"].append(preprocess)
        timings["inference_ms"].append(inference)
        timings["postprocess_ms"].append(postprocess)
        timings["end_to_end_ms"].append(preprocess + inference + postprocess)
        predictions[row["sample_id"]] = prediction
        raw.append({"sample_id": row["sample_id"], "predictions": prediction, "speed_ms": speed})
    return raw, predictions, timings, time.perf_counter() - started


def _aggregate_result(recipe, rows, dataset_root, raw, predictions, timings, wall):
    frames = [
        {
            "sample_id": row["sample_id"],
            "truth": _truth(dataset_root / row["label_relative_path"], input_size=recipe.input_size),
            "predictions": predictions.get(row["sample_id"], []),
        }
        for row in rows
    ]
    metrics = threshold_metrics(frames, recipe.confidence_threshold)
    classes = metrics["per_class"]
    tp = sum(value["tp"] for value in classes.values())
    fp = sum(value["fp"] for value in classes.values())
    fn = sum(value["fn"] for value in classes.values())
    environment, environment_identity = runtime_environment()
    result = {
        "sandbox_experiment_result_schema_version": 1,
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
        "experiment_id": recipe.experiment_id,
        "partition": recipe.partition,
        "input_size": recipe.input_size,
        "frame_skip_interval": recipe.frame_skip_interval,
        "selection_algorithm": recipe.selection_algorithm,
        "confidence_threshold": recipe.confidence_threshold,
        "frame_counts": {
            "source": len(rows), "inferred": len(raw),
            "skipped": len(rows) - len(raw),
            "labelled": sum(bool(frame["truth"]) for frame in frames),
        },
        "metrics": {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "per_class_recall": {name: value["recall"] for name, value in classes.items()},
            "small_object_recall": metrics["small_object_recall"],
            "no_target_false_positive_rate": metrics["no_target_false_positive_rate"],
        },
        "timing": {name: timing_summary(values) for name, values in timings.items()},
        "resources": {
            "wall_time_seconds": wall,
            "throughput_fps": len(raw) / max(wall, 1e-12),
            "peak_rss_bytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "runtime_environment": environment,
        "runtime_environment_identity": environment_identity,
        "warmup_frame_count": min(WARMUP_FRAME_COUNT, len(raw)),
        "metric_scope": "diagnostic_fixed_threshold_no_ap",
    }
    return result


def run_recipe(project_root, recipe_path):
    recipe, output, dataset_root, model_path, rows = _load_context(project_root, recipe_path)
    result_path = output / "result.json"
    if result_path.exists():
        raise ValueError("sandbox experiment result already exists")
    status = {
        "sandbox_experiment_status_schema_version": 1,
        "state": "running",
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
    }
    write_json(output / "status.json", status)
    try:
        with ordered_replay_sources(rows, dataset_root) as (source_root, paths):
            raw, predictions, timings, wall = _execute_model(
                model_path, rows, paths, source_root, recipe
            )
        _write_jsonl(output / "raw_predictions.jsonl", raw)
        result = _aggregate_result(
            recipe, rows, dataset_root, raw, predictions, timings, wall
        )
        result["raw_predictions_sha256"] = file_sha256(output / "raw_predictions.jsonl")
        result["result_identity_sha256"] = object_sha256(result)
        write_json(result_path, result)
        status.update({
            "state": "complete",
            "result_identity_sha256": result["result_identity_sha256"],
        })
        write_json(output / "status.json", status)
        return result
    except Exception as error:
        status.update({"state": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(output / "status.json", status)
        raise


def inspect_result(path):
    path = Path(path)
    result = json.loads(path.read_text())
    supplied = result.pop("result_identity_sha256", None)
    if supplied != object_sha256(result):
        raise ValueError("sandbox experiment result identity mismatch")
    result["result_identity_sha256"] = supplied
    return {"valid": True, "path": str(path), "result": result}
