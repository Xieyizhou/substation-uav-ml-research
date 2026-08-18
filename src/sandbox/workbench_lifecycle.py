"""Inspect and selectively re-run completed workbench stages."""

from __future__ import annotations

import json
from pathlib import Path

from src.sandbox.workbench_comparison import compare_with_baseline
from src.sandbox.workbench_models import WorkbenchExperimentRecipe
from src.sandbox.workbench_receipt import materialize_workbench_receipt
from src.sandbox.workbench_runner import (
    evaluate_workbench_model,
    replay_workbench_model,
    update_workbench_status,
)


def inspect_workbench_run(path):
    root = Path(path)
    result = {"run_root": str(root)}
    for name in ("recipe", "status", "view", "validation",
                 "onnx_equivalence", "replay", "comparison", "receipt"):
        candidate = root / f"{name}.json"
        result[name] = json.loads(candidate.read_text()) if candidate.is_file() else None
    return result


def list_workbench_runs(runs_root):
    root = Path(runs_root)
    return [inspect_workbench_run(path) for path in sorted(root.iterdir())
            if path.is_dir() and (path / "recipe.json").is_file()] if root.is_dir() else []


def _recipe(root):
    path = Path(root) / "recipe.json"
    return WorkbenchExperimentRecipe.from_record(json.loads(path.read_text()))


def validate_workbench_run(root):
    root, recipe = Path(root), _recipe(root)
    best, view = root / "training/weights/best.pt", root / "view"
    if not best.is_file() or not (view / "dataset.yaml").is_file():
        raise FileNotFoundError("workbench validation requires best.pt and its view")
    update_workbench_status(root, recipe, state="running", stage="validation", progress=0.0)
    result, _ = evaluate_workbench_model(best, view, root, recipe)
    update_workbench_status(
        root, recipe, state="complete", stage="validation_complete", progress=1.0
    )
    return result


def replay_workbench_run(project_root, root):
    root, recipe = Path(root), _recipe(root)
    size = recipe.parameters["imgsz"]
    onnx, view = root / f"model/model_{size}.onnx", root / "view"
    gate = json.loads((root / "onnx_equivalence.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    if not gate.get("passed") or not onnx.is_file():
        raise ValueError("workbench replay requires a passed ONNX equivalence gate")
    threshold = validation["confidence_evaluation"]["selected"]["threshold"]
    update_workbench_status(root, recipe, state="running", stage="replay", progress=0.0)
    result = replay_workbench_model(onnx, view, root, recipe, threshold)
    comparison = compare_with_baseline(Path(project_root), view, root, recipe, result)
    best = root / "training/weights/best.pt"
    receipt = materialize_workbench_receipt(
        Path(project_root), root, recipe, best, onnx, gate, result, comparison,
    )
    update_workbench_status(
        root, recipe, state="complete", stage="complete", progress=1.0,
        eta_seconds=0, error=None, failure_code=None,
        checkpoint_available=(root / "training/weights/last.pt").is_file(),
    )
    return {"replay": result, "receipt": receipt}
