import errno
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from src.cli.visual import build_parser
from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.ml.visual_identity import DatasetIdentity, class_order_identity
from src.ml.visual_heldout_view import _heldout_dataset_yaml
from src.ml.visual_training_identity import TrainingViewIdentity
from src.ml.visual_training_view import (
    evenly_select,
    materialize_training_view,
    proportional_quotas,
    select_partition,
)
from src.ml.visual_yolo_dataset import link_image, yolo_label_text
from src.ml.visual_yolo_training import load_training_config
from src.ml.visual_yolo_evaluation import (
    _standard_metrics,
    collect_predictions,
    evaluate_yolo,
)


HASH = "a" * 64


def annotation(sample, recording, split, class_name=None, sequence=1):
    objects = []
    if class_name:
        objects.append(
            {
                "class_name": class_name,
                "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
                "validation_status": "validated",
            }
        )
    return {
        "sample_id": sample,
        "recording_id": recording,
        "annotation": {
            "sequence_number": sequence,
            "annotation_status": (
                "labelled" if class_name else "verified_no_target"
            ),
            "mission_phase": "approach",
            "image_width": 1920,
            "image_height": 1080,
            "objects": objects,
        },
        "split": split,
    }


class SamplingTests(unittest.TestCase):
    def test_heldout_yaml_supports_test_but_disables_training_splits(self):
        content = _heldout_dataset_yaml("/tmp/heldout")
        self.assertIn("test: images/heldout_test\n", content)
        self.assertIn("train: disabled/heldout_not_for_training\n", content)
        self.assertIn("val: disabled/heldout_not_for_validation\n", content)

    def test_standard_metrics_enable_confusion_matrix_collection(self):
        class Box:
            mp = 0.9
            mr = 0.8
            map50 = 0.85
            map = 0.75
            ap_class_index = [0, 1, 2, 3]
            p = [0.9] * 4
            r = [0.8] * 4
            ap50 = [0.85] * 4
            ap = [0.75] * 4

        class Matrix:
            matrix = type("Values", (), {"tolist": lambda self: [[1.0]]})()

        metrics = types.SimpleNamespace(
            box=Box(),
            names=dict(enumerate(EQUIPMENT_CLASSES)),
            confusion_matrix=Matrix(),
        )
        model = types.SimpleNamespace(val=Mock(return_value=metrics))
        yolo = Mock(return_value=model)
        module = types.SimpleNamespace(YOLO=yolo)
        with patch.dict("sys.modules", {"ultralytics": module}):
            result = _standard_metrics(
                "model.pt", "dataset.yaml", split="test", device="cpu", imgsz=640
            )
        self.assertTrue(model.val.call_args.kwargs["plots"])
        self.assertFalse(model.val.call_args.kwargs["rect"])
        self.assertEqual(result["confusion_matrix"], [[1.0]])

    def test_prediction_collection_uses_static_square_preprocessing(self):
        model = types.SimpleNamespace(predict=Mock(return_value=[]))
        module = types.SimpleNamespace(YOLO=Mock(return_value=model))
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"ultralytics": module}
        ):
            self.assertEqual(
                collect_predictions(
                    "model.pt", directory, "full_validation",
                    device="cpu", imgsz=640,
                ),
                [],
            )
        self.assertFalse(model.predict.call_args.kwargs["rect"])
        self.assertEqual(model.predict.call_args.kwargs["batch"], 1)

    def test_largest_remainder_is_exact_stable_and_bounded(self):
        first = proportional_quotas({"b": 3, "a": 7}, 6)
        second = proportional_quotas({"a": 7, "b": 3}, 6)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 6)
        self.assertLessEqual(first["a"], 7)
        self.assertLessEqual(first["b"], 3)

    def test_even_selection_preserves_end_to_end_coverage(self):
        rows = [
            annotation(f"s-{index}", "r", "train", "transformer", index)
            for index in range(10)
        ]
        selected = evenly_select(rows, 3)
        self.assertEqual(
            [row["annotation"]["sequence_number"] for row in selected],
            [1, 5, 8],
        )

    def test_class_and_negative_caps_are_independent(self):
        rows = []
        for index in range(8):
            rows.append(
                annotation(
                    f"p-{index}",
                    f"r-{index % 2}",
                    "train",
                    "transformer",
                    index,
                )
            )
            rows.append(
                annotation(f"n-{index}", f"r-{index % 2}", "train", None, index)
            )
        selected = select_partition(rows, class_cap=3, negative_cap=2)
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            sum(bool(row["annotation"]["objects"]) for row in selected), 3
        )


class YoloDatasetTests(unittest.TestCase):
    def test_no_target_writes_an_empty_label(self):
        self.assertEqual(
            yolo_label_text(annotation("s", "r", "train")["annotation"]),
            "",
        )

    def test_label_uses_locked_class_order(self):
        text = yolo_label_text(
            annotation("s", "r", "train", "capacitor_bank")["annotation"]
        )
        self.assertTrue(text.startswith("2 "))

    def test_cross_device_link_falls_back_to_relative_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            destination = root / "nested/image.png"
            source.write_bytes(b"png")
            with patch(
                "src.ml.visual_yolo_dataset.os.link",
                side_effect=OSError(errno.EXDEV, "cross-device"),
            ):
                mode = link_image(source, destination)
            self.assertEqual(mode, "symlink")
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"png")


class TrainingIdentityTests(unittest.TestCase):
    def test_identity_detects_changed_membership(self):
        values = {
            "source_development_dataset_identity": HASH,
            "sampling_algorithm": "test-v1",
            "sampling_seed": 7,
            "train_membership_sha256": "b" * 64,
            "validation_membership_sha256": "c" * 64,
            "full_validation_membership_sha256": "d" * 64,
            "labels_manifest_sha256": "e" * 64,
            "class_order_identity": class_order_identity(),
            "train_frame_count": 5,
            "validation_frame_count": 2,
            "full_validation_frame_count": 4,
            "train_class_counts": {name: 1 for name in EQUIPMENT_CLASSES},
            "validation_class_counts": {name: 1 for name in EQUIPMENT_CLASSES},
            "train_no_target_count": 1,
            "validation_no_target_count": 1,
        }
        first = TrainingViewIdentity(**values)
        second = TrainingViewIdentity(
            **{**values, "train_membership_sha256": "f" * 64}
        )
        self.assertNotEqual(
            first.training_view_identity_sha256,
            second.training_view_identity_sha256,
        )
        self.assertEqual(
            TrainingViewIdentity.from_record(first.to_record()), first
        )
        with self.assertRaisesRegex(ValueError, "locked class order"):
            TrainingViewIdentity(
                **{**values, "train_class_counts": {"transformer": 1}}
            )


class MaterializationTests(unittest.TestCase):
    def _source(self, collection, recording, name):
        path = collection / "recordings" / recording / "frames" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload-{recording}-{name}".encode())
        return path

    def _write_collection(self, collection, include_test=False):
        identity_root = collection / "identity"
        identity_root.mkdir(parents=True)
        dataset = DatasetIdentity(
            dataset_name="development",
            dataset_version="v1",
            dataset_role="development",
            recording_schema_version=2,
            annotation_schema_version=1,
            decoder_configuration_id=HASH,
            recording_manifest_sha256="b" * 64,
            scenario_manifest_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
            ordered_frame_count=10,
            labelled_frame_count=8,
            source_recording_ids=("train-r", "validation-r"),
            scenario_ids=("train-s", "validation-s"),
            map_ids=("training",),
            seed_ids=(1, 2),
            class_order_identity=class_order_identity(),
            annotation_manifest_sha256="e" * 64,
        )
        write_json(identity_root / "development_dataset_identity.json", dataset.to_record())
        memberships, annotations = [], []
        index = 0
        for split in ("train", "validation"):
            for class_name in (*EQUIPMENT_CLASSES, None):
                recording = f"{split}-r"
                sample = f"{recording}:f-{index}"
                source = self._source(collection, recording, f"{index}.png")
                memberships.append(
                    {
                        "sample_id": sample,
                        "recording_id": recording,
                        "split": split,
                        "payload_relative_path": f"frames/{index}.png",
                        "payload_sha256": file_sha256(source),
                    }
                )
                annotations.append(
                    {
                        "sample_id": sample,
                        "recording_id": recording,
                        "annotation": annotation(
                            sample, recording, split, class_name, index
                        )["annotation"],
                    }
                )
                index += 1
        if include_test:
            memberships[0]["split"] = "test"
        for path, rows in (
            (identity_root / "development_membership.jsonl", memberships),
            (identity_root / "development_annotations.jsonl", annotations),
        ):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    def test_materialization_is_split_safe_and_does_not_copy_payload_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection, output = root / "collection", root / "view"
            self._write_collection(collection)
            result = materialize_training_view(collection, output)
            identity = result["identity"]
            self.assertEqual(identity["train_frame_count"], 5)
            self.assertEqual(identity["validation_frame_count"], 5)
            self.assertEqual(identity["full_validation_frame_count"], 5)
            image = next((output / "images/train").iterdir())
            self.assertGreaterEqual(image.stat().st_nlink, 2)
            self.assertIn("test: images/full_validation", (output / "dataset.yaml").read_text())

    def test_non_development_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection, output = root / "collection", root / "view"
            self._write_collection(collection, include_test=True)
            with self.assertRaisesRegex(ValueError, "non-development split"):
                materialize_training_view(collection, output)

    def test_training_view_rejects_a_different_sampling_seed(self):
        with self.assertRaisesRegex(ValueError, "sampling seed 7"):
            materialize_training_view("unused", "unused", seed=8)


class TrainingCliTests(unittest.TestCase):
    def test_cli_exposes_training_and_heldout_gates(self):
        parser = build_parser()
        args = parser.parse_args(["train-yolo", "--smoke"])
        self.assertTrue(args.smoke)
        heldout = parser.parse_args(
            ["heldout-view-materialize", "--package", "model", "--output", "test"]
        )
        self.assertEqual(heldout.command, "heldout-view-materialize")
        static = parser.parse_args(
            [
                "static-replay-materialize", "--package", "model",
                "--dataset", "test", "--output", "results",
            ]
        )
        self.assertEqual(static.command, "static-replay-materialize")

    def test_frozen_training_configuration_is_valid(self):
        config = load_training_config(
            "config/perception/visual_yolo11n_baseline.json"
        )
        self.assertEqual(config["batch"], 8)
        self.assertEqual(config["imgsz"], 640)
        self.assertFalse(config["amp"])

    def test_heldout_evaluation_requires_package_access_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "access receipt"):
                evaluate_yolo(
                    model,
                    root,
                    root / "result.json",
                    partition="heldout_test",
                )

    def test_heldout_uses_frozen_threshold_without_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"model")
            receipt = {
                "canonical_model_sha256": file_sha256(model),
                "frozen_confidence_threshold": 0.42,
                "heldout_dataset_identity_sha256": "b" * 64,
                "membership_sha256": "c" * 64,
            }
            write_json(root / "identity/heldout_access_receipt.json", receipt)
            frames = [
                {
                    "truth": [],
                    "predictions": [],
                    "sample_id": "sample",
                }
            ]
            output = root / "result.json"
            with patch(
                "src.ml.visual_yolo_evaluation._formal_commit",
                return_value="commit",
            ), patch(
                "src.ml.visual_yolo_evaluation._standard_metrics",
                return_value={"mAP50_95": 0.5},
            ), patch(
                "src.ml.visual_yolo_evaluation.collect_predictions",
                return_value=frames,
            ), patch(
                "src.ml.visual_yolo_evaluation.select_confidence_threshold"
            ) as search:
                result = evaluate_yolo(
                    model,
                    root,
                    output,
                    partition="heldout_test",
                )
            search.assert_not_called()
            self.assertEqual(
                result["confidence_evaluation"]["frozen"]["threshold"],
                0.42,
            )
            self.assertEqual(result["prediction_batch_size"], 1)
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluate_yolo(
                    model,
                    root,
                    output,
                    partition="heldout_test",
                )

    def test_full_validation_binds_training_view_and_selects_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "best.pt"
            model.write_bytes(b"weights")
            identity = TrainingViewIdentity(
                source_development_dataset_identity="a" * 64,
                sampling_algorithm="test",
                sampling_seed=7,
                train_membership_sha256="b" * 64,
                validation_membership_sha256="c" * 64,
                full_validation_membership_sha256="d" * 64,
                labels_manifest_sha256="e" * 64,
                class_order_identity=class_order_identity(),
                train_frame_count=4,
                validation_frame_count=4,
                full_validation_frame_count=4,
                train_class_counts={name: 1 for name in EQUIPMENT_CLASSES},
                validation_class_counts={name: 1 for name in EQUIPMENT_CLASSES},
                train_no_target_count=0,
                validation_no_target_count=0,
            )
            write_json(
                root / "identity/training_view_identity.json", identity.to_record()
            )
            frames = [{"sample_id": "sample", "truth": [], "predictions": []}]
            with patch(
                "src.ml.visual_yolo_evaluation._formal_commit",
                return_value="evaluation-commit",
            ), patch(
                "src.ml.visual_yolo_evaluation._standard_metrics",
                return_value={"mAP50_95": 0.5},
            ), patch(
                "src.ml.visual_yolo_evaluation.collect_predictions",
                return_value=frames,
            ):
                result = evaluate_yolo(
                    model,
                    root,
                    root / "result.json",
                    partition="full_validation",
                )
            self.assertEqual(
                result["dataset_provenance"]["training_view_identity_sha256"],
                identity.training_view_identity_sha256,
            )
            self.assertEqual(result["evaluation_code_commit_sha"], "evaluation-commit")
            self.assertEqual(result["prediction_batch_size"], 16)
            self.assertEqual(len(result["confidence_evaluation"]["candidates"]), 71)


if __name__ == "__main__":
    unittest.main()
