"""Read-only summaries of non-blind sandbox visual experiments."""

from __future__ import annotations

import json

from src.sandbox.experiment_recipe import inspect_recipe
from src.sandbox.experiment_runner import inspect_result


STATES = frozenset({"ready", "running", "complete", "failed"})


def _summary(directory):
    recipe = inspect_recipe(directory / "recipe.json")["recipe"]
    status = json.loads((directory / "status.json").read_text())
    state = status.get("state")
    if state not in STATES:
        raise ValueError("unsupported sandbox experiment state")
    if status.get("recipe_identity_sha256") != recipe["recipe_identity_sha256"]:
        raise ValueError("sandbox experiment status references a different recipe")
    result = None
    result_path = directory / "result.json"
    if result_path.is_file():
        result = inspect_result(result_path)["result"]
        if result["recipe_identity_sha256"] != recipe["recipe_identity_sha256"]:
            raise ValueError("sandbox result references a different recipe")
    if state == "complete" and result is None:
        raise ValueError("completed sandbox experiment lacks a result")
    if (
        state == "complete"
        and status.get("result_identity_sha256") != result["result_identity_sha256"]
    ):
        raise ValueError("sandbox experiment status references a different result")
    return {
        "experiment_id": recipe["experiment_id"],
        "state": state,
        "partition": recipe["partition"],
        "input_size": recipe["input_size"],
        "frame_skip_interval": recipe["frame_skip_interval"],
        "source_frame_count": recipe["source_frame_count"],
        "inference_frame_count": recipe["inference_frame_count"],
        "confidence_threshold": recipe["confidence_threshold"],
        "recipe_identity_sha256": recipe["recipe_identity_sha256"],
        "result_identity_sha256": None if result is None else result["result_identity_sha256"],
        "metrics": None if result is None else result["metrics"],
        "timing": None if result is None else result["timing"]["end_to_end_ms"],
        "resources": None if result is None else result["resources"],
        "error": status.get("error") if state == "failed" else None,
    }


def experiment_summaries(config):
    root = config.sandbox_experiments_root
    if not root.is_dir():
        return []
    rows = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        try:
            rows.append(_summary(directory))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            rows.append({
                "experiment_id": directory.name,
                "state": "invalid",
                "error": str(error),
            })
    return rows[:100]
