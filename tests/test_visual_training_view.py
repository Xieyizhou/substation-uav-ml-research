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
from src.vision.contracts.identity import DatasetIdentity, class_order_identity
from src.vision.evaluation.heldout_view import (
    _heldout_dataset_yaml,
    load_heldout_source,
)
from src.vision.evaluation.paired_heldout import (
    _paired_bootstrap,
    evaluate_paired_heldout,
    materialize_paired_heldout_view,
)
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.training.view import (
    evenly_select,
    materialize_training_view,
    proportional_quotas,
    select_partition,
)
from src.vision.training.yolo_dataset import link_image, yolo_label_text
from src.vision.training.yolo_training import load_training_config
from src.vision.evaluation.yolo_evaluation import (
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

    def test_even_selection_supports_hard_example_source_frame_ids(self):
        rows = [
            {
                "sample_id": f"hard-{index}",
                "source_frame_id": f"medium-4301-{index:08d}",
            }
            for index in range(10)
        ]
        selected = evenly_select(list(reversed(rows)), 3)
        self.assertEqual(
            [row["source_frame_id"] for row in selected],
            ["medium-4301-00000001", "medium-4301-00000005", "medium-4301-00000008"],
        )

    def test_paired_bootstrap_is_deterministic_and_recording_based(self):
        counts = {}
        for candidate, true_positives in (("v1", 1), ("v2", 2)):
            counts[candidate] = {}
            for recording in ("a", "b"):
                counts[candidate][recording] = {
                    name: {
                        "tp": true_positives,
                        "fp": 0,
                        "fn": 2 - true_positives,
                    }
                    for name in EQUIPMENT_CLASSES
                }
        first = _paired_bootstrap(counts)
        self.assertEqual(first, _paired_bootstrap(counts))
        self.assertEqual(first["unit"], "recording")
        self.assertEqual(first["repetitions"], 2000)
        self.assertEqual(first["probability_v2_improves"], 1.0)

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
                "src.vision.training.yolo_dataset.os.link",
                side_effect=OSError(errno.EXDEV, "cross-device"),
            ):
                mode = link_image(source, destination)
            self.assertEqual(mode, "symlink")
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"png")

    def test_partition_rematerialization_removes_stale_files(self):
        from src.vision.training.yolo_dataset import materialize_yolo_partition

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "output/images/train/stale.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            (root / "output/labels/train").mkdir(parents=True)
            materialize_yolo_partition("train", [], root / "source", root / "output")
            self.assertFalse(stale.exists())


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
        with_validation = TrainingViewIdentity(
            **{**values, "source_validation_dataset_identity": "f" * 64}
        )
        self.assertNotEqual(
            first.training_view_identity_sha256,
            with_validation.training_view_identity_sha256,
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

    def test_v2_heldout_contract_accepts_blind_but_not_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity_root = root / "identity"
            identity_root.mkdir()
            identity = DatasetIdentity(
                dataset_name="blind",
                dataset_version="visual-multiscenario-png-v2",
                dataset_role="held_out_test",
                recording_schema_version=2,
                annotation_schema_version=1,
                decoder_configuration_id=HASH,
                recording_manifest_sha256="b" * 64,
                scenario_manifest_sha256="c" * 64,
                split_manifest_sha256="d" * 64,
                ordered_frame_count=1,
                labelled_frame_count=0,
                source_recording_ids=("r",),
                scenario_ids=("s",),
                map_ids=("m",),
                seed_ids=(1,),
                class_order_identity=class_order_identity(),
                annotation_manifest_sha256="e" * 64,
            )
            write_json(
                identity_root / "held_out_test_dataset_identity.json",
                identity.to_record(),
            )
            membership = {
                "sample_id": "sample",
                "recording_id": "r",
                "split": "blind",
            }
            annotation_row = {
                "sample_id": "sample",
                "recording_id": "r",
                "annotation": {},
            }
            membership_path = identity_root / "held_out_test_membership.jsonl"
            annotation_path = identity_root / "held_out_test_annotations.jsonl"
            membership_path.write_text(json.dumps(membership) + "\n")
            annotation_path.write_text(json.dumps(annotation_row) + "\n")
            loaded, rows = load_heldout_source(root)
            self.assertEqual(loaded.dataset_version, identity.dataset_version)
            self.assertEqual(rows[0]["split"], "blind")
            membership["split"] = "test"
            membership_path.write_text(json.dumps(membership) + "\n")
            with self.assertRaisesRegex(ValueError, "non-blind split"):
                load_heldout_source(root)

    def test_paired_materialization_binds_both_frozen_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "view"
            heldout = types.SimpleNamespace(
                dataset_version="visual-multiscenario-png-v2",
                dataset_identity_sha256="d" * 64,
                to_record=lambda: {"dataset": "blind"},
            )
            candidates = [
                {
                    "package_root": f"/{name}",
                    "package_identity_sha256": char * 64,
                    "canonical_model_relative_path": "onnx/model_640.onnx",
                    "canonical_model_sha256": char.upper() * 64,
                    "frozen_confidence_threshold": threshold,
                }
                for name, char, threshold in (("v1", "a", 0.65), ("v2", "b", 0.37))
            ]
            partition = {
                "membership_sha256": "c" * 64,
                "frame_count": 10,
                "link_modes": {"hardlink": 10},
            }
            with patch(
                "src.vision.evaluation.paired_heldout._candidate",
                side_effect=candidates,
            ), patch(
                "src.vision.evaluation.paired_heldout.load_heldout_source",
                return_value=(heldout, []),
            ), patch(
                "src.vision.evaluation.paired_heldout.materialize_yolo_partition",
                return_value=partition,
            ):
                result = materialize_paired_heldout_view(
                    "collection", "v1", "v2", output
                )
            receipt = result["receipt"]
            self.assertEqual(receipt["candidate_order"], ["v1", "v2"])
            self.assertEqual(receipt["candidates"]["v1"]["frozen_confidence_threshold"], 0.65)
            self.assertEqual(receipt["candidates"]["v2"]["frozen_confidence_threshold"], 0.37)


class TrainingCliTests(unittest.TestCase):
    def test_cli_exposes_training_and_heldout_gates(self):
        parser = build_parser()
        args = parser.parse_args(["train-yolo", "--smoke"])
        self.assertTrue(args.smoke)
        heldout = parser.parse_args(
            ["heldout-view-materialize", "--package", "model", "--output", "test"]
        )
        self.assertEqual(heldout.command, "heldout-view-materialize")
        paired = parser.parse_args(
            [
                "paired-heldout-materialize",
                "--collection-root", "collection",
                "--v1-package", "v1",
                "--v2-package", "v2",
                "--output", "blind",
            ]
        )
        self.assertEqual(paired.command, "paired-heldout-materialize")
        paired_evaluate = parser.parse_args(
            ["paired-heldout-evaluate", "--dataset", "blind", "--output", "out"]
        )
        self.assertEqual(paired_evaluate.command, "paired-heldout-evaluate")
        static = parser.parse_args(
            [
                "static-replay-materialize", "--package", "model",
                "--dataset", "test", "--output", "results",
            ]
        )
        self.assertEqual(static.command, "static-replay-materialize")
        static_run = parser.parse_args(
            [
                "static-replay-run", "--input", "run", "--package", "model",
                "--dataset", "test", "--condition", "one",
            ]
        )
        self.assertEqual(static_run.condition_ids, ["one"])

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
                "src.vision.evaluation.yolo_evaluation._formal_commit",
                return_value="commit",
            ), patch(
                "src.vision.evaluation.yolo_evaluation._standard_metrics",
                return_value={"mAP50_95": 0.5},
            ), patch(
                "src.vision.evaluation.yolo_evaluation.collect_predictions",
                return_value=frames,
            ), patch(
                "src.vision.evaluation.yolo_evaluation.select_confidence_threshold"
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

    def test_paired_heldout_uses_fixed_models_thresholds_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = {}
            candidates = {}
            for name, threshold in (("v1", 0.65), ("v2", 0.37)):
                package = root / name
                model = package / "onnx/model_640.onnx"
                model.parent.mkdir(parents=True)
                model.write_bytes(name.encode())
                models[name] = model
                candidates[name] = {
                    "package_root": str(package),
                    "package_identity_sha256": name[1] * 64,
                    "canonical_model_relative_path": "onnx/model_640.onnx",
                    "canonical_model_sha256": file_sha256(model),
                    "frozen_confidence_threshold": threshold,
                }
            dataset = root / "dataset"
            membership = dataset / "identity/heldout_test_membership.jsonl"
            membership.parent.mkdir(parents=True)
            membership.write_text(
                json.dumps(
                    {
                        "image_relative_path": "images/heldout_test/sample.png",
                        "recording_id": "recording",
                    }
                )
                + "\n"
            )
            receipt = {
                "paired_heldout_access_schema_version": 1,
                "candidate_order": ["v1", "v2"],
                "heldout_dataset_identity_sha256": "d" * 64,
                "membership_sha256": "e" * 64,
                "frame_count": 1,
                "canonical_input_size": 640,
                "candidates": candidates,
            }
            receipt["paired_heldout_access_identity_sha256"] = object_sha256(receipt)
            write_json(
                dataset / "identity/paired_heldout_access_receipt.json", receipt
            )
            truth = [{"class_name": "transformer", "bbox": [0, 0, 10, 10]}]
            v1_frames = [{"sample_id": "sample", "truth": truth, "predictions": []}]
            v2_frames = [
                {
                    "sample_id": "sample",
                    "truth": truth,
                    "predictions": [
                        {
                            "class_name": "transformer",
                            "bbox": [0, 0, 10, 10],
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
            with patch(
                "src.vision.evaluation.paired_heldout._formal_commit",
                return_value="commit",
            ), patch(
                "src.vision.evaluation.paired_heldout._standard_metrics",
                side_effect=[{"mAP50_95": 0.1}, {"mAP50_95": 0.5}],
            ), patch(
                "src.vision.evaluation.paired_heldout.collect_predictions",
                side_effect=[v1_frames, v2_frames],
            ):
                output = root / "results"
                result = evaluate_paired_heldout(dataset, output)
            self.assertEqual(result["candidate_order"], ["v1", "v2"])
            self.assertEqual(result["comparison"]["mAP50_95_delta_v2_minus_v1"], 0.4)
            self.assertGreater(result["comparison"]["macro_f1_delta_v2_minus_v1"], 0)
            self.assertTrue((output / "predictions/v1.jsonl").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluate_paired_heldout(dataset, output)

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
                "src.vision.evaluation.yolo_evaluation._formal_commit",
                return_value="evaluation-commit",
            ), patch(
                "src.vision.evaluation.yolo_evaluation._standard_metrics",
                return_value={"mAP50_95": 0.5},
            ), patch(
                "src.vision.evaluation.yolo_evaluation.collect_predictions",
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
