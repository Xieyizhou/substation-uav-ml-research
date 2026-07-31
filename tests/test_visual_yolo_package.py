import json
from pathlib import Path
import tempfile
import unittest

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.visual_identity import ModelIdentity, class_order_identity
from src.ml.visual_yolo_package import (
    EXPORT_SIZES,
    preprocessing_identity,
    validate_yolo_package,
)


HASH = "a" * 64


class VisualYoloPackageTests(unittest.TestCase):
    def _package(self, root):
        weights = root / "weights/best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")
        weight_hash = file_sha256(weights)
        preprocessing_records, model_records, exports = {}, {}, {}
        for size in EXPORT_SIZES:
            model_file = root / "onnx" / f"model_{size}.onnx"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_bytes(f"onnx-{size}".encode())
            model_hash = file_sha256(model_file)
            preprocessing = preprocessing_identity(size)
            model = ModelIdentity(
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
                weights_sha256=weight_hash,
                model_file_sha256=model_hash,
                model_package_schema_version=1,
                runtime_version_identity="test",
                training_dataset_identity=HASH,
                training_code_commit_sha="abc123",
                export_configuration_identity="b" * 64,
                onnx_opset=19,
            )
            key = str(size)
            preprocessing_records[key] = preprocessing.to_record()
            model_records[key] = model.to_record()
            exports[key] = {
                "path": f"onnx/model_{size}.onnx",
                "sha256": model_hash,
                "configuration": {"input_size": size},
            }
        write_json(root / "preprocessing_identities.json", preprocessing_records)
        write_json(root / "model_identities.json", model_records)
        write_json(
            root / "full_validation_results.json",
            {"partition": "full_validation"},
        )
        manifest = {
            "visual_model_package_schema_version": 1,
            "architecture": "yolo11n",
            "weights": {"path": "weights/best.pt", "sha256": weight_hash},
            "exports": exports,
            "training_view_identity_sha256": HASH,
            "full_validation_results_sha256": file_sha256(
                root / "full_validation_results.json"
            ),
            "model_identity_sha256": {
                size: record["model_identity_sha256"]
                for size, record in model_records.items()
            },
        }
        manifest["package_identity_sha256"] = object_sha256(manifest)
        write_json(root / "manifest.json", manifest)

    def test_valid_package_binds_each_export_and_preprocessing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._package(root)
            self.assertTrue(validate_yolo_package(root)["valid"])

    def test_changed_onnx_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._package(root)
            (root / "onnx/model_416.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "ONNX hash mismatch"):
                validate_yolo_package(root)


if __name__ == "__main__":
    unittest.main()
