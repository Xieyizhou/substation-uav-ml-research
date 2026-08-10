import json
from pathlib import Path
import sys
from types import ModuleType
import tempfile
import unittest
from unittest.mock import patch

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.contracts.identity import ModelIdentity, class_order_identity
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.evaluation.yolo_package import (
    EXPORT_SIZES,
    export_yolo_package,
    preprocessing_identity,
    validate_yolo_package,
)


HASH = "a" * 64


class _FakeYolo:
    def __init__(self, weights):
        self.weights = Path(weights)

    def export(self, *, imgsz, **_):
        output = self.weights.with_suffix(".onnx")
        output.write_bytes(f"onnx-{imgsz}".encode())
        return str(output)


class VisualYoloPackageTests(unittest.TestCase):
    def _ultralytics(self):
        module = ModuleType("ultralytics")
        module.YOLO = _FakeYolo
        return patch.dict(sys.modules, {"ultralytics": module})

    def _export_inputs(self, root):
        identity = TrainingViewIdentity(
            source_development_dataset_identity="1" * 64,
            sampling_algorithm="test",
            sampling_seed=7,
            train_membership_sha256="2" * 64,
            validation_membership_sha256="3" * 64,
            full_validation_membership_sha256="4" * 64,
            labels_manifest_sha256="5" * 64,
            class_order_identity=class_order_identity(),
            train_frame_count=4,
            validation_frame_count=4,
            full_validation_frame_count=4,
            train_class_counts={name: 1 for name in EQUIPMENT_CLASSES},
            validation_class_counts={name: 1 for name in EQUIPMENT_CLASSES},
            train_no_target_count=0,
            validation_no_target_count=0,
        )
        identity_path = root / "training_view_identity.json"
        write_json(identity_path, identity.to_record())
        weights = root / "source/best.pt"
        weights.parent.mkdir()
        weights.write_bytes(b"weights")
        provenance = root / "source/training_provenance.json"
        write_json(
            provenance,
            {
                "training_view_identity_sha256": identity.training_view_identity_sha256,
                "training_code_commit_sha": "commit",
            },
        )
        validation = root / "validation.json"
        write_json(
            validation,
            {
                "partition": "full_validation",
                "model_sha256": file_sha256(weights),
                "dataset_provenance": {
                    "training_view_identity_sha256": identity.training_view_identity_sha256
                },
                "confidence_evaluation": {"selected": {"threshold": 0.42}},
            },
        )
        return weights, identity_path, provenance, validation

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
            {
                "partition": "full_validation",
                "model_sha256": weight_hash,
                "confidence_evaluation": {"selected": {"threshold": 0.42}},
            },
        )
        write_json(root / "training_provenance.json", {"environment": {}})
        write_json(root / "training_view_identity.json", {"identity": HASH})
        gates = {}
        for size in EXPORT_SIZES:
            key = str(size)
            path = root / "equivalence" / f"onnx_equivalence_{size}.json"
            write_json(
                path,
                {
                    "passed": True,
                    "equivalence_code_commit_sha": "gate-commit",
                    "input_size": size,
                    "pt_model_sha256": weight_hash,
                    "onnx_model_sha256": exports[key]["sha256"],
                },
            )
            gates[key] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
        write_json(
            root / "package_status.json",
            {
                "visual_model_package_status_schema_version": 1,
                "status": "finalized",
            },
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
            "supporting_artifacts": {
                name: file_sha256(root / name)
                for name in ("training_provenance.json", "training_view_identity.json")
            },
            "frozen_confidence_threshold": 0.42,
            "equivalence_gates": gates,
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

    def test_changed_supporting_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._package(root)
            write_json(root / "training_provenance.json", {"environment": {"changed": True}})
            with self.assertRaisesRegex(ValueError, "supporting artifact hash mismatch"):
                validate_yolo_package(root)

    def test_missing_or_failed_equivalence_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._package(root)
            manifest = json.loads((root / "manifest.json").read_text())
            del manifest["equivalence_gates"]["416"]
            identity = manifest.pop("package_identity_sha256")
            self.assertIsNotNone(identity)
            manifest["package_identity_sha256"] = object_sha256(manifest)
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing ONNX equivalence"):
                validate_yolo_package(root)

    def test_export_writes_manifest_only_after_all_equivalence_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._export_inputs(root)

            def gate(pt, onnx, _, output, *, imgsz, device):
                result = {
                    "passed": imgsz != 416,
                    "input_size": imgsz,
                    "pt_model_sha256": file_sha256(pt),
                    "onnx_model_sha256": file_sha256(onnx),
                }
                write_json(output, result)
                return result

            output = root / "package"
            with self._ultralytics(), patch(
                "src.vision.evaluation.yolo_package.validate_onnx_equivalence",
                side_effect=gate,
            ):
                with self.assertRaisesRegex(ValueError, "input size 416"):
                    export_yolo_package(
                        *inputs,
                        output,
                        equivalence_dataset=root,
                    )
            self.assertFalse((output / "manifest.json").exists())
            status = json.loads((output / "package_status.json").read_text())
            self.assertEqual(status["status"], "staging_failed")


if __name__ == "__main__":
    unittest.main()
