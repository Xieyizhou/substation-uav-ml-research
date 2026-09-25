"""Integrity checks for finalized visual model packages."""

import json
from pathlib import Path
from src.ml.artifacts import file_sha256, object_sha256
from src.vision.contracts.identity import ModelIdentity, PreprocessingIdentity
from src.vision.evaluation.yolo_package_v3 import validate_v3_gate

EXPORT_SIZES = (320, 416, 640)


def validate_yolo_package(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    supplied = manifest.pop("package_identity_sha256", None)
    if supplied != object_sha256(manifest):
        raise ValueError("visual model package identity mismatch")
    if file_sha256(root / manifest["weights"]["path"]) != manifest["weights"]["sha256"]:
        raise ValueError("visual model package weights hash mismatch")
    if (
        file_sha256(root / "full_validation_results.json")
        != manifest["full_validation_results_sha256"]
    ):
        raise ValueError("full-validation results hash mismatch")
    for path, expected in manifest.get("supporting_artifacts", {}).items():
        if file_sha256(root / path) != expected:
            raise ValueError(f"supporting artifact hash mismatch: {path}")
    validate_v3_gate(root, manifest.get("real_domain_v3_gate"))
    validation = json.loads((root / "full_validation_results.json").read_text())
    threshold = manifest.get("frozen_confidence_threshold")
    selected = validation.get("confidence_evaluation", {}).get("selected", {})
    if threshold != selected.get("threshold"):
        raise ValueError("frozen confidence threshold mismatch")
    status = json.loads((root / "package_status.json").read_text())
    if status.get("status") != "finalized":
        raise ValueError("visual model package is not finalized")
    model_records = json.loads((root / "model_identities.json").read_text())
    preprocessing_records = json.loads(
        (root / "preprocessing_identities.json").read_text()
    )
    for size in EXPORT_SIZES:
        key = str(size)
        export = manifest["exports"][key]
        if file_sha256(root / export["path"]) != export["sha256"]:
            raise ValueError(f"ONNX hash mismatch for input size {size}")
        model = ModelIdentity.from_record(model_records[key])
        preprocessing = PreprocessingIdentity.from_record(
            preprocessing_records[key]
        )
        if model.model_file_sha256 != export["sha256"]:
            raise ValueError(f"model identity mismatch for input size {size}")
        if (
            model.preprocessing_configuration_id
            != preprocessing.preprocessing_configuration_id
        ):
            raise ValueError(f"preprocessing identity mismatch for input size {size}")
        gate_record = manifest.get("equivalence_gates", {}).get(key)
        if not isinstance(gate_record, dict):
            raise ValueError(f"missing ONNX equivalence gate for input size {size}")
        gate_path = root / gate_record.get("path", "")
        if file_sha256(gate_path) != gate_record.get("sha256"):
            raise ValueError(f"ONNX equivalence hash mismatch for input size {size}")
        gate = json.loads(gate_path.read_text())
        if (
            gate.get("passed") is not True
            or not isinstance(gate.get("equivalence_code_commit_sha"), str)
            or gate.get("input_size") != size
            or gate.get("onnx_model_sha256") != export["sha256"]
            or gate.get("pt_model_sha256") != manifest["weights"]["sha256"]
        ):
            raise ValueError(f"invalid ONNX equivalence gate for input size {size}")
    return {
        **manifest,
        "package_identity_sha256": supplied,
        "valid": True,
        "path": str(root),
    }
