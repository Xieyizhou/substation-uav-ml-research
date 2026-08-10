import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.ml.dataset import ResearchSample
from src.ml.dataset_builder import build_dataset_manifest
from src.ml.model_package import create_model_package, validate_model_package
from src.ml.predictions import evaluate_predictions, predict_dataset
from src.ml.research_recorder import ResearchDatasetWriter
from src.ml.train_lidar import (
    _is_better_checkpoint,
    _validate_training_label_coverage,
    train,
)


ML_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "onnx", "onnxruntime")
)


def sample(split, map_id, seed, index, risk):
    distance = {"clear": 8.0, "warning": 2.0, "danger": 0.5}[risk]
    values = [8.0] * 72
    values[30 + index % 12] = distance
    traversability = tuple(min(value / 5.0, 1.0) for value in values)
    return ResearchSample(
        scenario_id=f"{split}-{index}",
        split=split,
        map_id=map_id,
        seed=seed + index,
        timestamp_s=float(index),
        ranges_m=tuple(values),
        range_max_m=10.0,
        velocity_ned_m_s=(0.5, 0.0, 0.0),
        pose_ned_m=(0.0, 0.0, -1.5),
        yaw_deg=0.0,
        risk_label=risk,
        traversability=traversability,
        recommended_direction_deg=-30.0 + index * 5.0,
    )


@unittest.skipUnless(
    ML_RUNTIME_AVAILABLE, "optional PyTorch/ONNX research stack is not installed"
)
class EndToEndTrainingTests(unittest.TestCase):
    def test_checkpoint_selection_prioritizes_macro_f1_over_total_loss(self):
        self.assertTrue(
            _is_better_checkpoint(
                {"macro_f1": 0.7, "loss": 5.0}, best_f1=0.6, best_loss=1.0
            )
        )
        self.assertFalse(
            _is_better_checkpoint(
                {"macro_f1": 0.5, "loss": 0.1}, best_f1=0.6, best_loss=1.0
            )
        )
        self.assertTrue(
            _is_better_checkpoint(
                {"macro_f1": 0.6, "loss": 0.9}, best_f1=0.6, best_loss=1.0
            )
        )

    def test_training_rejects_missing_risk_classes_by_default(self):
        samples = [sample("train", "simple", 2001, 0, "clear")]
        with self.assertRaisesRegex(ValueError, "warning, danger"):
            _validate_training_label_coverage(samples)
        _validate_training_label_coverage(samples, allow_incomplete=True)

    def test_one_epoch_train_export_package_predict_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            samples_path = dataset / "samples.jsonl"
            with ResearchDatasetWriter(samples_path) as writer:
                for index, risk in enumerate(("clear", "warning", "danger") * 2):
                    writer.append(sample("train", "simple", 2001, index, risk))
                for index, risk in enumerate(("clear", "warning", "danger")):
                    writer.append(
                        sample("validation", "complex", 2041, index, risk)
                    )
                    writer.append(sample("test", "extreme", 2051, index, risk))
            manifest = build_dataset_manifest(samples_path)
            manifest_path = dataset / "dataset_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            package = root / "model"
            result = train(
                samples_path,
                package,
                epochs=1,
                batch_size=3,
                patience=1,
                seed=9,
                model_id="tiny-test",
            )
            create_model_package(
                package,
                onnx_path=result["model"],
                dataset_manifest_path=manifest_path,
                training_history_path=result["history"],
                offline_metrics_path=result["metrics"],
                model_id="tiny-test",
            )
            self.assertEqual(
                validate_model_package(package)["model_id"], "tiny-test"
            )
            predictions = root / "predictions.jsonl"
            predict_dataset(package / "model.onnx", samples_path, predictions)
            metrics = evaluate_predictions(predictions)
            self.assertEqual(metrics["latency"]["count"], 3)
            self.assertIn("macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()
