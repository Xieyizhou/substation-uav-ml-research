"""Materialize and execute identity-bound static visual replay conditions."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import resource
import time

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.ml.visual_benchmark import (
    TIMING_STAGES,
    VisualBenchmarkCondition,
    VisualBenchmarkResult,
)
from src.ml.visual_benchmark_matrix import validate_static_benchmark_directory
from src.ml.visual_detection_metrics import threshold_metrics
from src.ml.visual_identity import DatasetIdentity, ModelIdentity, PreprocessingIdentity
from src.ml.visual_pilot import _read_jsonl, _write_jsonl
from src.ml.visual_static_source import (
    ordered_replay_sources,
    static_predict_options,
    write_source_list,
)
from src.ml.visual_static_runtime import runtime_environment, timing_summary
from src.ml.visual_yolo_evaluation import _prediction_rows, _truth
from src.ml.visual_yolo_package import validate_yolo_package


def _clean_commit():
    commit = git_commit()
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("static visual replay requires a clean tracked worktree")
    return commit


def _load_inputs(package_root, heldout_root):
    package_root, heldout_root = Path(package_root), Path(heldout_root)
    package = validate_yolo_package(package_root)
    receipt = json.loads(
        (heldout_root / "identity/heldout_access_receipt.json").read_text()
    )
    if receipt.get("model_package_identity_sha256") != package[
        "package_identity_sha256"
    ]:
        raise ValueError("held-out receipt references a different model package")
    dataset = DatasetIdentity.from_record(
        json.loads(
            (heldout_root / "identity/held_out_test_dataset_identity.json").read_text()
        )
    )
    membership_path = heldout_root / "identity/heldout_test_membership.jsonl"
    rows = _read_jsonl(membership_path)
    if file_sha256(membership_path) != receipt["membership_sha256"]:
        raise ValueError("held-out replay membership hash mismatch")
    models = json.loads((package_root / "model_identities.json").read_text())
    preprocessing = json.loads(
        (package_root / "preprocessing_identities.json").read_text()
    )
    return package, receipt, dataset, rows, models, preprocessing


def materialize_static_replay(
    package_root, heldout_root, benchmark_root, output_root
):
    matrix = validate_static_benchmark_directory(benchmark_root)
    package, receipt, dataset, rows, models, preprocessing = _load_inputs(
        package_root, heldout_root
    )
    output_root = Path(output_root)
    if (output_root / "manifest.json").exists():
        raise ValueError("static replay matrix is already materialized")
    commit = _clean_commit()
    conditions = []
    for template in matrix["templates"]:
        size = str(template["input_size"])
        model = ModelIdentity.from_record(models[size])
        preprocess = PreprocessingIdentity.from_record(preprocessing[size])
        condition = VisualBenchmarkCondition(
            condition_id=template["template_id"],
            dataset_identity_sha256=dataset.dataset_identity_sha256,
            decoder_configuration_id=dataset.decoder_configuration_id,
            preprocessing_configuration_id=preprocess.preprocessing_configuration_id,
            model_identity_sha256=model.model_identity_sha256,
            input_width=model.input_width,
            input_height=model.input_height,
            inference_policy=template["inference_policy"],
            frame_skip_interval=template["frame_skip_interval"],
            confidence_threshold=receipt["frozen_confidence_threshold"],
            target_inference_rate_hz=None,
            roi_mode="disabled",
            batch_size=1,
            warmup_frame_count=template["warmup_frame_count"],
            measured_frame_count=len(rows),
            runtime_backend=model.runtime_backend,
            device_identity=f"cpu-{platform.machine()}",
            precision=model.precision,
            deadline_definition="offline_no_deadline",
            random_seed=None,
            software_commit_sha=commit,
        )
        condition.validate_references(dataset, model, preprocess)
        path = output_root / "conditions" / f"{condition.condition_id}.json"
        write_json(path, condition.to_record())
        conditions.append(
            {
                "condition_id": condition.condition_id,
                "path": path.relative_to(output_root).as_posix(),
                "condition_identity_sha256": condition.condition_identity_sha256,
            }
        )
    manifest = {
        "static_replay_manifest_schema_version": 1,
        "benchmark_id": matrix["benchmark_id"],
        "package_identity_sha256": package["package_identity_sha256"],
        "heldout_access_identity_sha256": receipt[
            "heldout_access_identity_sha256"
        ],
        "dataset_identity_sha256": dataset.dataset_identity_sha256,
        "membership_sha256": receipt["membership_sha256"],
        "software_commit_sha": commit,
        "conditions": conditions,
    }
    manifest["static_replay_manifest_sha256"] = object_sha256(manifest)
    write_json(output_root / "manifest.json", manifest)
    return manifest


def _run_condition(condition, rows, ordered_paths, source_root, heldout_root,
                   model_path, output_root):
    from ultralytics import YOLO
    selected = rows[:: condition.frame_skip_interval]
    selected_paths = ordered_paths[:: condition.frame_skip_interval]
    model = YOLO(str(model_path))
    options = static_predict_options(condition)
    warmup = selected_paths[: condition.warmup_frame_count]
    if warmup:
        source = write_source_list(source_root, "warmup", warmup)
        list(model.predict(str(source), **options))
    timings = {stage: [] for stage in TIMING_STAGES}
    predictions, raw = {}, []
    started = time.perf_counter()
    source = write_source_list(source_root, condition.condition_id, selected_paths)
    results = model.predict(str(source), **options)
    for row, result in zip(selected, results, strict=True):
        prediction = _prediction_rows(result)
        predictions[row["sample_id"]] = prediction
        speed = result.speed
        preprocess = float(speed.get("preprocess", 0.0))
        inference = float(speed.get("inference", 0.0))
        postprocess = float(speed.get("postprocess", 0.0))
        timings["preprocess_ms"].append(preprocess)
        timings["backend_call_ms"].append(inference)
        timings["inference_ms"].append(inference)
        timings["postprocess_ms"].append(postprocess)
        timings["end_to_end_ms"].append(preprocess + inference + postprocess)
        raw.append(
            {"sample_id": row["sample_id"], "predictions": prediction, "speed_ms": speed}
        )
    wall = time.perf_counter() - started
    frames = []
    for row in rows:
        frames.append(
            {
                "sample_id": row["sample_id"],
                "truth": _truth(
                    Path(heldout_root) / row["label_relative_path"],
                    input_size=condition.input_width,
                ),
                "predictions": predictions.get(row["sample_id"], []),
            }
        )
    metrics = threshold_metrics(frames, condition.confidence_threshold)
    totals = metrics["per_class"].values()
    tp, fp, fn = (sum(row[key] for row in totals) for key in ("tp", "fp", "fn"))
    raw_path = output_root / "raw" / f"{condition.condition_id}.jsonl"
    _write_jsonl(raw_path, raw)
    raw_manifest = {
        "raw_result_manifest_schema_version": 1,
        "condition_identity_sha256": condition.condition_identity_sha256,
        "inference_frame_count": len(raw),
        "raw_predictions_sha256": file_sha256(raw_path),
    }
    manifest_path = output_root / "raw" / f"{condition.condition_id}.manifest.json"
    write_json(manifest_path, raw_manifest)
    environment, environment_id = runtime_environment()
    inferred = len(raw)
    visual = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "mAP50": None,
        "mAP50_95": None,
        "per_class_recall": {
            name: row["recall"] for name, row in metrics["per_class"].items()
        },
        "small_object_recall": metrics["small_object_recall"],
        "no_target_false_positive_rate": metrics["no_target_false_positive_rate"],
    }
    resources = {
        "wall_time_seconds": wall,
        "throughput_fps": inferred / max(wall, 1e-12),
        "peak_rss_bytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "energy_joules": None,
    }
    availability = {
        name: value is not None for name, value in {**visual, **resources}.items()
    }
    result = VisualBenchmarkResult(
        condition_identity_sha256=condition.condition_identity_sha256,
        dataset_identity_sha256=condition.dataset_identity_sha256,
        model_identity_sha256=condition.model_identity_sha256,
        decoder_configuration_id=condition.decoder_configuration_id,
        preprocessing_configuration_id=condition.preprocessing_configuration_id,
        software_commit_sha=condition.software_commit_sha,
        runtime_environment_identity=environment_id,
        started=True,
        completion_status="completed",
        frame_counts={
            "total": len(rows), "decoded": len(rows), "inferred": inferred,
            "skipped": len(rows) - inferred, "failed_decode": 0,
            "failed_inference": 0,
            "labelled": sum(bool(frame["truth"]) for frame in frames),
        },
        timing_summaries={stage: timing_summary(values) for stage, values in timings.items()},
        scheduling_summaries={
            "requested_inference_frames": len(selected),
            "completed_inference_frames": inferred,
            "deadline_misses": 0,
            "dropped_or_unavailable_frames": 0,
        },
        visual_metrics=visual,
        resource_metrics=resources,
        metric_availability=availability,
        unavailable_metrics=tuple(name for name, value in availability.items() if not value),
        failure_codes=(),
        raw_result_artifact_manifest_sha256=file_sha256(manifest_path),
    )
    return result, environment


def run_static_replay(matrix_root, package_root, heldout_root):
    matrix_root = Path(matrix_root)
    manifest = json.loads((matrix_root / "manifest.json").read_text())
    supplied = manifest.pop("static_replay_manifest_sha256", None)
    if supplied != object_sha256(manifest):
        raise ValueError("static replay manifest identity mismatch")
    package, _, dataset, rows, models, preprocessing = _load_inputs(
        package_root, heldout_root
    )
    completed = []
    commit = _clean_commit()
    with ordered_replay_sources(rows, heldout_root) as (source_root, paths):
        for item in manifest["conditions"]:
            condition = VisualBenchmarkCondition.from_record(
                json.loads((matrix_root / item["path"]).read_text())
            )
            if condition.software_commit_sha != commit:
                raise ValueError("static replay condition references different code")
            size = str(condition.input_width)
            model = ModelIdentity.from_record(models[size])
            preprocess = PreprocessingIdentity.from_record(preprocessing[size])
            condition.validate_references(dataset, model, preprocess)
            result_path = matrix_root / "results" / f"{condition.condition_id}.json"
            if result_path.exists():
                existing = VisualBenchmarkResult.from_record(
                    json.loads(result_path.read_text())
                )
                existing.validate_references(condition, dataset, model, preprocess)
                completed.append(condition.condition_id)
                continue
            model_path = Path(package_root) / package["exports"][size]["path"]
            result, environment = _run_condition(
                condition, rows, paths, source_root, heldout_root,
                model_path, matrix_root,
            )
            result.validate_references(condition, dataset, model, preprocess)
            write_json(result_path, result.to_record())
            write_json(matrix_root / "runtime_environment.json", environment)
            completed.append(condition.condition_id)
    return {"completed_condition_ids": completed, "condition_count": len(completed)}
