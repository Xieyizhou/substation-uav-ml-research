import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, write_json
from src.ml.visual_identity import class_order_identity
from src.ml.visual_training_identity import TrainingViewIdentity
from src.ml.visual_yolo_training import train_yolo


class _FakeYolo:
    calls = []

    def __init__(self, weights):
        self.weights = weights
        self.trainer = None

    def train(self, **kwargs):
        self.calls.append(kwargs)
        root = Path(kwargs["project"]) / kwargs["name"]
        weights = root / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        best, last = weights / "best.pt", weights / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        (root / "results.csv").write_text("epoch,metric\n0,1\n")
        self.trainer = SimpleNamespace(best=best, last=last)
        return SimpleNamespace(results_dict={"mAP50-95": 0.5})


class VisualYoloTrainingTests(unittest.TestCase):
    def _dataset(self, root):
        identity = TrainingViewIdentity(
            source_development_dataset_identity="a" * 64,
            sampling_algorithm="test",
            sampling_seed=7,
            train_membership_sha256="b" * 64,
            validation_membership_sha256="c" * 64,
            full_validation_membership_sha256="d" * 64,
            labels_manifest_sha256="e" * 64,
            class_order_identity=class_order_identity(),
            train_frame_count=10,
            validation_frame_count=5,
            full_validation_frame_count=8,
            train_class_counts={name: 1 for name in EQUIPMENT_CLASSES},
            validation_class_counts={name: 1 for name in EQUIPMENT_CLASSES},
            train_no_target_count=2,
            validation_no_target_count=1,
        )
        write_json(root / "identity/training_view_identity.json", identity.to_record())
        (root / "dataset.yaml").write_text("train: images/train\nval: images/val\n")

    def test_training_records_clean_commit_and_resolved_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "runs/baseline"
            self._dataset(dataset)
            actual_hash = file_sha256(Path("yolo11n.pt"))
            config = json.loads(
                Path("config/perception/visual_yolo11n_baseline.json").read_text()
            )
            config["pretrained_weights_sha256"] = actual_hash
            config_path = root / "config.json"
            write_json(config_path, config)
            _FakeYolo.calls.clear()
            with patch("ultralytics.YOLO", _FakeYolo), patch(
                "src.ml.visual_yolo_training._require_clean_commit",
                return_value="commit-sha",
            ):
                result = train_yolo(config_path, dataset, output)
            provenance = json.loads(
                Path(result["training_provenance"]).read_text()
            )
            self.assertEqual(
                provenance["training_code_commit_sha"], "commit-sha"
            )
            self.assertEqual(provenance["resolved_config"]["batch"], 8)
            self.assertEqual(_FakeYolo.calls[0]["device"], "mps")
            self.assertEqual(
                Path(_FakeYolo.calls[0]["project"]),
                output.resolve().parent,
            )
            self.assertTrue(
                Path(_FakeYolo.calls[0]["data"]).is_absolute()
            )

    def test_existing_checkpoint_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "runs/baseline"
            self._dataset(dataset)
            checkpoint = output / "weights/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"last")
            with patch(
                "src.ml.visual_yolo_training._require_clean_commit",
                return_value="commit-sha",
            ):
                with self.assertRaisesRegex(ValueError, "--resume"):
                    train_yolo(
                        "config/perception/visual_yolo11n_baseline.json",
                        dataset,
                        output,
                    )


if __name__ == "__main__":
    unittest.main()
