"""Prevent duplicate workbench execution and verify completed runs."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256


@contextmanager
def exclusive_workbench_run(run_root):
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("workbench experiment is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verified_completed_receipt(run_root, recipe):
    root = Path(run_root)
    path = root / "receipt.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("workbench_receipt_schema_version") != 1
        or receipt.get("experiment_id") != recipe.experiment_id
        or receipt.get("recipe_identity_sha256") != recipe.recipe_identity_sha256
        or receipt.get("passed") is not True
    ):
        raise ValueError("workbench completion receipt is invalid")
    identity = receipt.get("receipt_identity_sha256")
    identity_payload = dict(receipt)
    identity_payload.pop("receipt_identity_sha256", None)
    if identity != object_sha256(identity_payload):
        raise ValueError("workbench completion receipt identity changed")
    size = recipe.parameters["imgsz"]
    artifacts = {
        "best_weights_sha256": root / "training/weights/best.pt",
        "onnx_model_sha256": root / f"model/model_{size}.onnx",
        "validation_sha256": root / "validation.json",
        "equivalence_sha256": root / "onnx_equivalence.json",
    }
    if "training_efficiency_sha256" in receipt:
        artifacts["training_efficiency_sha256"] = root / "training_efficiency.json"
    for name in ("comparison", "replay"):
        if f"{name}_sha256" in receipt:
            artifacts[f"{name}_sha256"] = root / f"{name}.json"
    for field, artifact in artifacts.items():
        if not artifact.is_file() or receipt.get(field) != file_sha256(artifact):
            raise ValueError(f"workbench completed artifact changed: {artifact.name}")
    replay = json.loads((root / "replay.json").read_text(encoding="utf-8"))
    if (
        replay.get("replay_identity_sha256") != receipt.get("replay_identity_sha256")
        or replay.get("model_sha256") != receipt.get("onnx_model_sha256")
    ):
        raise ValueError("workbench replay identity changed")
    return receipt
