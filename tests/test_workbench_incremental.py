"""Regression checks for historical weights and one-pass replay."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.sandbox.workbench_models import WorkbenchExperimentRecipe
from src.sandbox.workbench_weights import pin_weights, resolve_initial_weights
from src.sandbox.workbench_runner import replay_workbench_model
from src.vision.evaluation.yolo_evaluation import collect_predictions


class IncrementalWorkbenchTests(unittest.TestCase):
    def test_freeze_depth_is_bounded_and_identity_bound(self):
        options = dict(experiment_id="freeze", dataset_id="data", preset="smoke",
                       dataset_identity_sha256="a"*64, pretrained_weights_sha256="b"*64)
        recipe = WorkbenchExperimentRecipe.create(**options, overrides={"freeze": 10})
        self.assertEqual(WorkbenchExperimentRecipe.from_record(recipe.to_record()).parameters["freeze"], 10)
        for value in (True, -1, 11, "10"):
            with self.assertRaisesRegex(ValueError, "freeze"):
                WorkbenchExperimentRecipe.create(**options, overrides={"freeze": value})

    def test_private_snapshot_reused_and_original_may_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "historical.pt"
            source.write_bytes(b"historical weights")
            runs = root / "outputs/sandbox/workbench/runs"
            initialization, digest = pin_weights(root, runs, source)
            again, _ = pin_weights(root, runs, source)
            self.assertEqual(again, initialization)
            snapshot = root / initialization["checkpoint_path"]
            self.assertFalse(snapshot.samefile(source))
            self.assertEqual(len(list(snapshot.parent.glob("*.pt"))), 1)
            source.write_bytes(b"edited original")
            recipe = WorkbenchExperimentRecipe.create(
                experiment_id="child", dataset_id="data", preset="smoke",
                dataset_identity_sha256="a" * 64,
                pretrained_weights_sha256=digest, initialization=initialization,
            )
            restored = WorkbenchExperimentRecipe.from_record(recipe.to_record())
            self.assertEqual(resolve_initial_weights(root, restored), snapshot.resolve())
            snapshot.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "weights changed"):
                resolve_initial_weights(root, restored)

    def test_legacy_recipe_keeps_its_original_identity(self):
        # Construct a v1 record without the new optional initialization field.
        recipe = WorkbenchExperimentRecipe.create(
            experiment_id="old", dataset_id="data", preset="smoke",
            dataset_identity_sha256="a" * 64, pretrained_weights_sha256="b" * 64,
        )
        record = recipe.to_record()
        self.assertNotIn("initialization", record)
        self.assertEqual(WorkbenchExperimentRecipe.from_record(record).to_record(), record)

    def test_checkpoint_cannot_resolve_outside_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            recipe = WorkbenchExperimentRecipe.create(
                experiment_id="child", dataset_id="data", preset="smoke",
                dataset_identity_sha256="a" * 64, pretrained_weights_sha256="b" * 64,
                initialization={"checkpoint_path": "../outside.pt"},
            )
            with self.assertRaisesRegex(ValueError, "inside the project"):
                resolve_initial_weights(temporary, recipe)

    def test_single_prediction_stream_supplies_metrics_and_timings(self):
        calls = []
        result = types.SimpleNamespace(path="sample.png", orig_shape=(20, 30), boxes=None)
        class FakeModel:
            def __init__(self, path):
                calls.append(("load", path))
            def predict(self, **options):
                calls.append(("predict", options))
                return iter([result])
        module = types.ModuleType("ultralytics")
        module.YOLO = FakeModel
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels/validation"
            labels.mkdir(parents=True)
            (labels / "sample.txt").write_text("")
            onnx = root / "model.onnx"
            onnx.write_bytes(b"onnx")
            recipe = types.SimpleNamespace(parameters={"imgsz": 320})
            with patch.dict("sys.modules", {"ultralytics": module}), patch(
                "src.sandbox.workbench_runner.runtime_environment", return_value=({}, "env")
            ):
                replay = replay_workbench_model(onnx, root, root, recipe, 0.4)
                plain = collect_predictions(onnx, root, "validation", device="cpu", imgsz=320)
            self.assertEqual([call[0] for call in calls], ["load", "predict", "load", "predict"])
            self.assertEqual(calls[1][1], calls[3][1])
            self.assertEqual(replay["frame_count"], len(plain))
            self.assertEqual(replay["successful_frame_count"], 1)
            self.assertEqual(replay["prediction_passes"], 1)


if __name__ == "__main__":
    unittest.main()
