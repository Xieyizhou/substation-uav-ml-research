"""Dependency-free demonstration of an identity-bound ML workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.ml.artifacts import git_commit, object_sha256, write_json


DEMO_SCHEMA_VERSION = 1
TRAINING = (
    (3.8, 0.08, "safe"), (3.1, 0.12, "safe"),
    (2.7, 0.18, "safe"), (2.3, 0.22, "safe"),
    (1.2, 0.64, "danger"), (0.9, 0.73, "danger"),
    (0.7, 0.82, "danger"), (0.5, 0.91, "danger"),
)
EVALUATION = (
    (3.5, 0.10, "safe"), (2.9, 0.16, "safe"),
    (2.5, 0.20, "safe"), (2.0, 0.28, "safe"),
    (1.4, 0.58, "danger"), (1.0, 0.70, "danger"),
    (0.8, 0.79, "danger"), (0.4, 0.94, "danger"),
)


def _centroids():
    result = {}
    for label in ("safe", "danger"):
        rows = [row for row in TRAINING if row[2] == label]
        result[label] = [
            sum(row[index] for row in rows) / len(rows) for index in (0, 1)
        ]
    return result


def _predict(features, centroids):
    distances = {
        label: sum((value - center[index]) ** 2 for index, value in enumerate(features))
        for label, center in centroids.items()
    }
    return min(distances, key=lambda label: (distances[label], label))


def _metrics(predictions):
    labels = ("safe", "danger")
    confusion = {
        truth: {predicted: 0 for predicted in labels} for truth in labels
    }
    for row in predictions:
        confusion[row["expected"]][row["predicted"]] += 1
    f1 = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in labels if truth != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    correct = sum(row["expected"] == row["predicted"] for row in predictions)
    return {"accuracy": correct / len(predictions), "macro_f1": sum(f1) / len(f1),
            "confusion": confusion}


def run_demo(project_root, output):
    root = Path(output)
    centroids = _centroids()
    recipe = {
        "demo_recipe_schema_version": DEMO_SCHEMA_VERSION,
        "workflow": "demo_contract",
        "dataset_role": "demo",
        "formal_evidence": False,
        "algorithm": "nearest_centroid_v1",
        "feature_order": ["minimum_distance_m", "obstacle_density"],
        "training_sample_count": len(TRAINING),
        "evaluation_sample_count": len(EVALUATION),
        "software_commit_sha": git_commit(project_root),
    }
    recipe["demo_recipe_identity_sha256"] = object_sha256(recipe)
    predictions = []
    for index, row in enumerate(EVALUATION):
        predicted = _predict(row[:2], centroids)
        predictions.append({"sample_id": f"demo-{index:02d}", "expected": row[2],
                            "predicted": predicted})
    result = {
        "demo_result_schema_version": DEMO_SCHEMA_VERSION,
        "recipe_identity_sha256": recipe["demo_recipe_identity_sha256"],
        "centroids": centroids,
        "predictions": predictions,
        "metrics": _metrics(predictions),
        "passed": all(row["expected"] == row["predicted"] for row in predictions),
        "scope_note": "Illustrative synthetic data; not formal research evidence.",
    }
    if not all(math.isfinite(value) for value in result["metrics"].values()
               if isinstance(value, float)):
        raise ValueError("demo metrics contain non-finite values")
    result["demo_result_identity_sha256"] = object_sha256(result)
    write_json(root / "demo_recipe.json", recipe)
    write_json(root / "demo_result.json", result)
    return {"recipe": recipe, "result": result, "output": str(root)}


def inspect_demo(path):
    root = Path(path)
    recipe = json.loads((root / "demo_recipe.json").read_text(encoding="utf-8"))
    result = json.loads((root / "demo_result.json").read_text(encoding="utf-8"))
    recipe_identity = recipe.pop("demo_recipe_identity_sha256", None)
    result_identity = result.pop("demo_result_identity_sha256", None)
    if recipe_identity != object_sha256(recipe):
        raise ValueError("demo recipe identity mismatch")
    if result_identity != object_sha256(result):
        raise ValueError("demo result identity mismatch")
    if result["recipe_identity_sha256"] != recipe_identity:
        raise ValueError("demo result references a different recipe")
    if recipe.get("formal_evidence") is not False or recipe.get("dataset_role") != "demo":
        raise ValueError("demo workflow cannot be formal evidence")
    return {"valid": True, "passed": result["passed"],
            "recipe_identity_sha256": recipe_identity,
            "result_identity_sha256": result_identity, "metrics": result["metrics"]}
