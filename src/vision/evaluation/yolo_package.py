"""Export and validate identity-bound visual YOLO model packages."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.contracts.identity import (
    ModelIdentity,
    PreprocessingIdentity,
    class_order_identity,
)
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.evaluation.onnx_gate import validate_onnx_equivalence
from src.vision.evaluation.yolo_package_v3 import bind_v3_gate, validate_v3_gate
EXPORT_SIZES = (320, 416, 640)
def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
def preprocessing_identity(size):
    return PreprocessingIdentity(
        target_input_width=size,
        target_input_height=size,
        resize_policy="aspect_ratio_preserving_resize",
        letterbox_policy="centered_stride32_padding",
        interpolation_method="opencv_linear",
        padding_value=(114, 114, 114),
        hwc_to_chw=True,
        rgb_bgr_policy="canonical_rgb8_to_backend_bgr8_then_backend_rgb",
        uint8_to_float=True,
        normalization_scale=255.0,
        mean=None,
        standard_deviation=None,
        batch_dimension_policy="fixed_batch_1",
        tensor_dtype="float32",
        contiguous_memory_policy="contiguous_chw",
        implementation="ultralytics-letterbox",
        implementation_version=_version("ultralytics"),
    )
def export_yolo_package(
    weights,
    training_view_identity_path,
    training_provenance_path,
    validation_results_path,
    output_root,
    *,
    equivalence_dataset,
    equivalence_device="cpu",
    v3_gate_path=None,
):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("visual export requires requirements-ml.txt") from error
    weights, output_root = Path(weights), Path(output_root)
    training_view = TrainingViewIdentity.from_record(
        json.loads(Path(training_view_identity_path).read_text())
    )
    provenance = json.loads(Path(training_provenance_path).read_text())
    if (
        provenance.get("training_view_identity_sha256")
        != training_view.training_view_identity_sha256
    ):
        raise ValueError("training provenance references a different training view")
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ValueError("visual model package output must be empty")
    status_path = output_root / "package_status.json"
    write_json(
        status_path,
        {"visual_model_package_status_schema_version": 1, "status": "staging"},
    )
    validation_results = json.loads(Path(validation_results_path).read_text())
    if validation_results.get("partition") != "full_validation":
        raise ValueError("model package requires full-validation results")
    if validation_results.get("visual_evaluation_schema_version", 1) >= 2 and (
        validation_results.get("postprocessing_consistency", {}).get("passed")
        is not True
    ):
        raise ValueError(
            "model package requires runtime/validator postprocessing consistency"
        )
    if validation_results.get("model_sha256") != file_sha256(weights):
        raise ValueError("full-validation results reference different weights")
    validation_dataset = validation_results.get("dataset_provenance", {})
    if (
        validation_dataset.get("training_view_identity_sha256")
        != training_view.training_view_identity_sha256
    ):
        raise ValueError("full-validation results reference a different dataset")
    selected = validation_results.get("confidence_evaluation", {}).get("selected")
    frozen_threshold = selected.get("threshold") if isinstance(selected, dict) else None
    if not isinstance(frozen_threshold, (int, float)):
        raise ValueError("full-validation results lack a selected threshold")
    weights_copy = output_root / "weights/best.pt"
    weights_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, weights_copy)
    weights_sha = file_sha256(weights_copy)
    preprocess_records, model_records, exports = {}, {}, {}
    model = YOLO(str(weights_copy))
    for size in EXPORT_SIZES:
        exported = Path(
            model.export(
                format="onnx",
                imgsz=size,
                batch=1,
                dynamic=False,
                simplify=True,
                opset=19,
                nms=False,
                device="cpu",
            )
        )
        destination = output_root / "onnx" / f"model_{size}.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, destination)
        preprocessing = preprocessing_identity(size)
        export_record = {
            "format": "onnx",
            "input_size": size,
            "batch": 1,
            "dynamic": False,
            "simplify": True,
            "opset": 19,
            "nms": False,
            "precision": "fp32",
        }
        model_identity = ModelIdentity(
            model_family="yolo",
            architecture_variant="yolo11n",
            task="object_detection",
            class_order_identity=class_order_identity(),
            input_width=size,
            input_height=size,
            input_pixel_format="rgb8",
            preprocessing_configuration_id=(
                preprocessing.preprocessing_configuration_id
            ),
            runtime_backend="ultralytics-onnxruntime",
            precision="fp32",
            weights_sha256=weights_sha,
            model_file_sha256=file_sha256(destination),
            model_package_schema_version=1,
            runtime_version_identity=(
                f"ultralytics:{_version('ultralytics')}|"
                f"onnxruntime:{_version('onnxruntime')}"
            ),
            training_dataset_identity=(
                training_view.training_view_identity_sha256
            ),
            training_code_commit_sha=provenance["training_code_commit_sha"],
            export_configuration_identity=object_sha256(export_record),
            onnx_opset=19,
        )
        preprocess_records[str(size)] = preprocessing.to_record()
        model_records[str(size)] = model_identity.to_record()
        exports[str(size)] = {
            "path": destination.relative_to(output_root).as_posix(),
            "sha256": file_sha256(destination),
            "configuration": export_record,
        }
    write_json(output_root / "preprocessing_identities.json", preprocess_records)
    write_json(output_root / "model_identities.json", model_records)
    shutil.copy2(training_view_identity_path, output_root / "training_view_identity.json")
    shutil.copy2(training_provenance_path, output_root / "training_provenance.json")
    shutil.copy2(validation_results_path, output_root / "full_validation_results.json")
    v3_gate = bind_v3_gate(v3_gate_path, weights, training_view, output_root)
    history_source = Path(training_provenance_path).parent / "results.csv"
    if history_source.is_file():
        shutil.copy2(history_source, output_root / "training_history.csv")
    gates = {}
    try:
        for size in EXPORT_SIZES:
            path = output_root / "equivalence" / f"onnx_equivalence_{size}.json"
            gate = validate_onnx_equivalence(
                weights_copy,
                output_root / exports[str(size)]["path"],
                equivalence_dataset,
                path,
                imgsz=size,
                device=equivalence_device,
            )
            if not gate.get("passed"):
                raise ValueError(f"ONNX equivalence failed for input size {size}")
            gates[str(size)] = {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(path),
            }
    except Exception:
        write_json(
            status_path,
            {
                "visual_model_package_status_schema_version": 1,
                "status": "staging_failed",
            },
        )
        raise
    manifest = {
        "visual_model_package_schema_version": 1,
        "architecture": "yolo11n",
        "weights": {
            "path": "weights/best.pt",
            "sha256": weights_sha,
        },
        "exports": exports,
        "training_view_identity_sha256": (
            training_view.training_view_identity_sha256
        ),
        "full_validation_results_sha256": file_sha256(output_root / "full_validation_results.json"),
        "supporting_artifacts": {
            name: file_sha256(output_root / name)
            for name in ("training_view_identity.json", "training_provenance.json",
                         "training_history.csv", "preprocessing_identities.json",
                         "model_identities.json")
            if (output_root / name).is_file()
        },
        "frozen_confidence_threshold": float(frozen_threshold),
        "equivalence_gates": gates,
        "model_identity_sha256": {
            size: record["model_identity_sha256"]
            for size, record in model_records.items()
        },
    }
    if v3_gate is not None:
        manifest["real_domain_v3_gate"] = v3_gate
    manifest["package_identity_sha256"] = object_sha256(manifest)
    write_json(
        status_path,
        {"visual_model_package_status_schema_version": 1, "status": "finalized"},
    )
    write_json(output_root / "manifest.json", manifest)
    return validate_yolo_package(output_root)


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
