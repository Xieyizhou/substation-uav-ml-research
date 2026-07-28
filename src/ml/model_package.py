"""Create and validate self-describing ONNX model packages."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json


MODEL_MANIFEST_VERSION = 1


def create_model_package(
    package_dir,
    *,
    onnx_path,
    dataset_manifest_path,
    training_history_path,
    offline_metrics_path,
    model_id=None,
    parent_model=None,
    training_parameters=None,
    root=None,
):
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    source_model = Path(onnx_path)
    destination_model = package_dir / "model.onnx"
    if source_model.resolve() != destination_model.resolve():
        shutil.copy2(source_model, destination_model)
    for source, name in (
        (training_history_path, "training_history.json"),
        (offline_metrics_path, "offline_metrics.json"),
    ):
        source = Path(source)
        destination = package_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    dataset = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    metrics = json.loads((package_dir / "offline_metrics.json").read_text(encoding="utf-8"))
    onnx_hash = file_sha256(destination_model)
    identity = {
        "onnx_sha256": onnx_hash,
        "dataset_id": dataset["dataset_id"],
        "parent_model": parent_model,
    }
    manifest = {
        "manifest_schema_version": MODEL_MANIFEST_VERSION,
        "model_id": model_id or f"lidar-cnn-{object_sha256(identity)[:12]}",
        "parent_model": parent_model,
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": dataset["data_sha256"],
        "network": "lidar-multitask-1d-cnn-v2",
        "preprocessing": "normalized-resampled-scan-v1",
        "training_parameters": training_parameters or {},
        "training_commit": git_commit(root),
        "onnx_sha256": onnx_hash,
        "input_contract": {
            "laser_scan": ["batch", 1, 360],
            "dtype": "float32",
            "normalization": "range/range_max",
        },
        "output_contract": {
            "risk_logits": ["batch", 3],
            "traversability": ["batch", 72],
            "direction_deg": ["batch", 1],
            "uncertainty": ["batch", 1],
        },
        "offline_metrics": metrics,
    }
    write_json(package_dir / "model_manifest.json", manifest)
    return manifest


def validate_model_package(package_dir):
    package_dir = Path(package_dir)
    required = (
        "model.onnx",
        "model_manifest.json",
        "training_history.json",
        "offline_metrics.json",
    )
    missing = [name for name in required if not (package_dir / name).is_file()]
    if missing:
        raise ValueError("model package is missing: " + ", ".join(missing))
    manifest = json.loads(
        (package_dir / "model_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("onnx_sha256") != file_sha256(package_dir / "model.onnx"):
        raise ValueError("model package ONNX hash mismatch")
    return manifest
