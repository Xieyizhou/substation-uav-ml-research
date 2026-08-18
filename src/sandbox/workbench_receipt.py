"""Materialize the final receipt for a development workbench run."""

from __future__ import annotations

from pathlib import Path

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json


def materialize_workbench_receipt(
    project_root, run_root, recipe, best, onnx, gate, replay, comparison,
):
    root = Path(run_root)
    receipt = {
        "workbench_receipt_schema_version": 1,
        "experiment_id": recipe.experiment_id,
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
        "dataset_identity_sha256": recipe.dataset_identity_sha256,
        "software_commit_sha": git_commit(project_root),
        "dataset_role": "development",
        "formal_evidence": False,
        "best_weights_sha256": file_sha256(best),
        "onnx_model_sha256": file_sha256(onnx),
        "validation_sha256": file_sha256(root / "validation.json"),
        "equivalence_sha256": file_sha256(root / "onnx_equivalence.json"),
        "replay_identity_sha256": replay["replay_identity_sha256"],
        "comparison_status": comparison["status"],
        "passed": bool(gate["passed"]),
    }
    receipt["receipt_identity_sha256"] = object_sha256(receipt)
    write_json(root / "receipt.json", receipt)
    return receipt
