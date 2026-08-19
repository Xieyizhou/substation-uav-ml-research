"""Read-only presentation records for the visual model workbench."""

from __future__ import annotations

import json
from pathlib import Path

from src.sandbox.workbench_datasets import list_workbench_datasets
from src.sandbox.workbench_lifecycle import inspect_workbench_run, list_workbench_runs
from src.sandbox.workbench_inference import list_verified_models
from src.ml.artifacts import object_sha256


def _run_summary(value):
    recipe, status = value.get("recipe") or {}, value.get("status") or {}
    validation, replay = value.get("validation") or {}, value.get("replay") or {}
    selected = validation.get("confidence_evaluation", {}).get("selected", {})
    standard = validation.get("standard_metrics", {})
    comparison = value.get("comparison") or {}
    return {
        "experiment_id": recipe.get("experiment_id"),
        "dataset_id": recipe.get("dataset_id"), "preset": recipe.get("preset"),
        "parameters": recipe.get("parameters", {}), "state": status.get("state"),
        "stage": status.get("stage"), "progress": status.get("progress"),
        "epoch": status.get("epoch"), "total_epochs": status.get("total_epochs"),
        "eta_seconds": status.get("eta_seconds"), "error": status.get("error"),
        "failure_code": status.get("failure_code"),
        "checkpoint_available": status.get("checkpoint_available", False),
        "metrics": {
            "map50_95": standard.get("mAP50_95"),
            "precision": standard.get("precision"), "recall": standard.get("recall"),
            "macro_f1": selected.get("macro_f1"),
            "small_object_recall": selected.get("small_object_recall"),
            "no_target_false_positive_rate": selected.get("no_target_false_positive_rate"),
        },
        "threshold": selected.get("threshold"), "timing": replay.get("timing"),
        "comparison": comparison.get("deltas"),
        "comparison_status": comparison.get("status"),
    }


def workbench_summary(config):
    enabled = config.profile == "development"
    if not enabled:
        return {"enabled": False, "datasets": [], "runs": [],
                "reason": "Model workbench requires Development profile."}
    datasets = list_workbench_datasets(
        config.project_root, config.workbench_datasets_root
    )
    runs = [_run_summary(row) for row in list_workbench_runs(config.workbench_runs_root)]
    return {
        "enabled": True, "datasets": datasets, "runs": runs,
        "verified_models": list_verified_models(config.workbench_runs_root),
        "inferences": _inference_summaries(config.workbench_inference_root),
        "readiness": {
            "pretrained_weights": (config.project_root / "yolo11n.pt").is_file(),
            "dataset_available": bool(datasets),
            "output_writable": _writable(config.workbench_root),
        },
    }


def _inference_summaries(root):
    values = []
    for path in sorted(Path(root).iterdir(), reverse=True) if Path(root).is_dir() else ():
        candidate = path / "result.json"
        if not candidate.is_file():
            continue
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
            identity = record.pop("inference_identity_sha256", None)
            if identity != object_sha256(record):
                continue
            record["inference_identity_sha256"] = identity
            values.append({
                "inference_id": record["inference_id"],
                "created_at": record["created_at"],
                "input_width": record["input_width"],
                "input_height": record["input_height"],
                "results": record["results"],
            })
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if len(values) == 12:
            break
    return values


def workbench_inference(config, inference_id):
    path = config.workbench_inference(inference_id) / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.pop("inference_identity_sha256", None)
    if identity != object_sha256(value):
        raise ValueError("workbench inference identity changed")
    value["inference_identity_sha256"] = identity
    return value


def _writable(path):
    parent = next((item for item in [Path(path), *Path(path).parents]
                   if item.exists()), None)
    return bool(parent and parent.is_dir() and parent.stat().st_mode & 0o200)


def workbench_run(config, experiment_id):
    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError("invalid workbench experiment identifier")
    root = config.workbench_runs_root / experiment_id
    if root.parent.resolve() != config.workbench_runs_root.resolve():
        raise ValueError("workbench run is outside the approved root")
    return inspect_workbench_run(root)
