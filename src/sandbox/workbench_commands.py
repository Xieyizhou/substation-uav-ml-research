"""Allow-listed operator commands for model workbench runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from src.sandbox.command_models import SandboxCommand
from src.sandbox.workbench_recipe import materialize_workbench_recipe


def _baseline(config):
    root = config.project_root / "models/equipment/visual-yolo11n-baseline-v1-package"
    return root if (root / "manifest.json").is_file() else None


def _timeout(preset):
    return {"smoke": 7_200.0, "quick": 86_400.0, "full": 259_200.0}[preset]


def _parameters(value):
    if not isinstance(value, dict):
        raise ValueError("workbench parameters must be an object")
    allowed = {"experiment_id", "dataset_id", "preset", "epochs", "patience",
               "imgsz", "batch", "workers", "device"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unsupported workbench fields: {sorted(unknown)}")
    return value


def build_workbench_command(config, action, parameters):
    if config.profile != "development":
        raise ValueError("model workbench is available only in development profile")
    if action == "workbench-dataset-import":
        values = parameters if isinstance(parameters, dict) else {}
        if set(values) not in ({"source", "dataset_id"},
                               {"source", "dataset_id", "class_map"}):
            raise ValueError("dataset import accepts source, dataset_id, and class_map")
        source = Path(str(values["source"])).expanduser().resolve()
        if not source.is_dir() or not (source / "dataset.yaml").is_file():
            raise ValueError("selected folder is not a YOLO dataset")
        mapping = values.get("class_map", {})
        if not isinstance(mapping, dict):
            raise ValueError("class_map must be an object")
        canonical = {"transformer", "switchgear", "capacitor_bank", "reactor"}
        if set(mapping.values()) - canonical:
            raise ValueError("class_map contains an unsupported target class")
        if mapping and (set(mapping.values()) != canonical or len(mapping) != 4):
            raise ValueError("class_map must map four distinct source IDs")
        mapping_args = tuple(
            value for source_id, target in sorted(mapping.items(), key=lambda row: int(row[0]))
            for value in ("--class-map", f"{int(source_id)}={target}")
        )
        return SandboxCommand(
            action, (
                sys.executable, "main.py", "sandbox", "--project-root",
                str(config.project_root), "--profile", "development",
                "workbench-dataset-import", "--source", str(source),
                "--dataset-id", str(values["dataset_id"]),
                *mapping_args,
            ), 7_200.0, workflow="visual_workbench_dataset_import",
            requires_runtime_idle=False,
        )
    values = _parameters(parameters)
    if action == "workbench-run":
        required = ("experiment_id", "dataset_id", "preset")
        if any(not values.get(name) for name in required):
            raise ValueError("workbench run requires experiment, dataset, and preset")
        overrides = {name: values[name] for name in
                     ("epochs", "patience", "imgsz", "batch", "workers", "device")
                     if name in values}
        recipe, run_root = materialize_workbench_recipe(
            config.project_root, config.workbench_datasets_root,
            config.workbench_runs_root, values["experiment_id"],
            values["dataset_id"], values["preset"], overrides,
            _baseline(config),
        )
        command = "workbench-run"
    elif action == "workbench-resume":
        experiment_id = values.get("experiment_id")
        if not experiment_id or set(values) != {"experiment_id"}:
            raise ValueError("workbench resume accepts only experiment_id")
        run_root = config.workbench_runs_root / experiment_id
        recipe_path = run_root / "recipe.json"
        if not (run_root / "training/weights/last.pt").is_file():
            raise ValueError("workbench run has no resumable checkpoint")
        recipe = json.loads(recipe_path.read_text())
        command = "workbench-resume"
    elif action in {"workbench-validate", "workbench-replay"}:
        experiment_id = values.get("experiment_id")
        if not experiment_id or set(values) != {"experiment_id"}:
            raise ValueError(f"{action} accepts only experiment_id")
        run_root = config.workbench_runs_root / experiment_id
        recipe_path = run_root / "recipe.json"
        recipe = json.loads(recipe_path.read_text())
        command = action
    else:
        return None
    recipe_path = run_root / "recipe.json"
    preset = recipe.preset if hasattr(recipe, "preset") else recipe["preset"]
    result_name = {
        "workbench-validate": "validation.json",
        "workbench-replay": "replay.json",
    }.get(command, "receipt.json")
    relative_result = (run_root / result_name).relative_to(config.project_root).as_posix()
    return SandboxCommand(
        action, (
            sys.executable, "main.py", "sandbox", "--project-root",
            str(config.project_root), "--profile", "development", command,
            *(('--recipe', str(recipe_path)) if command in {
                "workbench-run", "workbench-resume"
            } else ('--input', str(run_root))),
        ), _timeout(preset), workflow="visual_workbench",
        expected_outputs=(relative_result,), requires_runtime_idle=True,
    )
