"""Create and validate allow-listed visual workbench experiment recipes."""

from __future__ import annotations

import json
from pathlib import Path
import re

from src.ml.artifacts import file_sha256, write_json
from src.sandbox.workbench_datasets import resolve_workbench_dataset
from src.sandbox.workbench_models import WorkbenchExperimentRecipe


IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def materialize_workbench_recipe(
    project_root, imported_root, runs_root, experiment_id, dataset_id, preset,
    overrides=None, baseline_package=None,
    *, initial_weights=None, parent_experiment_id=None,
):
    if not IDENTIFIER.fullmatch(str(experiment_id)):
        raise ValueError("experiment id must use lowercase letters, numbers, and hyphens")
    dataset = resolve_workbench_dataset(project_root, imported_root, dataset_id)
    output = Path(runs_root) / experiment_id
    if output.exists():
        raise ValueError("workbench experiment id already exists")
    initialization = None
    if initial_weights is not None and parent_experiment_id is not None:
        raise ValueError("choose initial weights or a parent experiment, not both")
    if parent_experiment_id is not None:
        if not IDENTIFIER.fullmatch(str(parent_experiment_id)):
            raise ValueError("invalid parent experiment identifier")
        from src.sandbox.workbench_inference import verified_model
        parent = Path(runs_root) / parent_experiment_id
        model = verified_model(parent)
        initial_weights = parent / "training/weights/best.pt"
    weights = Path(project_root) / "yolo11n.pt"
    if initial_weights is not None:
        from src.sandbox.workbench_weights import pin_weights
        initialization, weights_hash = pin_weights(project_root, runs_root, initial_weights)
        if parent_experiment_id is not None:
            if weights_hash != model["best_weights_sha256"]:
                raise ValueError("parent checkpoint changed during recipe creation")
            initialization.update(parent_experiment_id=parent_experiment_id,
                                  parent_receipt_identity_sha256=model["receipt_identity_sha256"],
                                  parent_threshold=model["threshold"])
        weights = Path(project_root) / initialization["checkpoint_path"]
    if not weights.is_file():
        raise FileNotFoundError(weights)
    baseline_identity = None
    if baseline_package:
        manifest = json.loads((Path(baseline_package) / "manifest.json").read_text())
        baseline_identity = manifest["package_identity_sha256"]
    recipe = WorkbenchExperimentRecipe.create(
        experiment_id=experiment_id,
        dataset_id=dataset.dataset_id,
        dataset_identity_sha256=dataset.dataset_identity_sha256,
        preset=preset,
        pretrained_weights_sha256=file_sha256(weights),
        initialization=initialization,
        overrides=overrides or {},
        baseline_package_identity_sha256=baseline_identity,
    )
    write_json(output / "recipe.json", recipe.to_record())
    write_json(output / "status.json", {
        "workbench_status_schema_version": 1,
        "experiment_id": experiment_id,
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
        "state": "created", "stage": "preflight", "progress": 0.0,
        "epoch": 0, "total_epochs": recipe.parameters["epochs"],
        "eta_seconds": None, "error": None, "failure_code": None,
        "checkpoint_available": False,
    })
    return recipe, output


def inspect_workbench_recipe(path):
    return WorkbenchExperimentRecipe.from_record(
        json.loads(Path(path).read_text(encoding="utf-8"))
    ).to_record()
