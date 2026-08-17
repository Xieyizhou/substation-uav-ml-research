"""Read-only presentation records for the visual model workbench."""

from __future__ import annotations

from pathlib import Path

from src.sandbox.workbench_datasets import list_workbench_datasets
from src.sandbox.workbench_lifecycle import inspect_workbench_run, list_workbench_runs


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
        "readiness": {
            "pretrained_weights": (config.project_root / "yolo11n.pt").is_file(),
            "dataset_available": bool(datasets),
            "output_writable": _writable(config.workbench_root),
        },
    }


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
