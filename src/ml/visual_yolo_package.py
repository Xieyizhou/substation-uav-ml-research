"""Export and validate identity-bound visual YOLO model packages."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.visual_identity import (
    ModelIdentity,
    PreprocessingIdentity,
    class_order_identity,
)
from src.ml.visual_training_identity import TrainingViewIdentity


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
    if (output_root / "manifest.json").exists():
        raise ValueError("visual model package is already frozen")
    validation_results = json.loads(Path(validation_results_path).read_text())
    if validation_results.get("partition") != "full_validation":
        raise ValueError("model package requires full-validation results")
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
    history_source = Path(training_provenance_path).parent / "results.csv"
    if history_source.is_file():
        shutil.copy2(history_source, output_root / "training_history.csv")
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
        "full_validation_results_sha256": file_sha256(
            output_root / "full_validation_results.json"
        ),
        "model_identity_sha256": {
            size: record["model_identity_sha256"]
            for size, record in model_records.items()
        },
    }
    manifest["package_identity_sha256"] = object_sha256(manifest)
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
    return {
        **manifest,
        "package_identity_sha256": supplied,
        "valid": True,
        "path": str(root),
    }
