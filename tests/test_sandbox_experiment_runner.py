import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ml.artifacts import write_json
from src.sandbox.experiment_recipe import ExperimentRecipe
from src.sandbox.experiment_runner import (
    _approved_recipe_path,
    inspect_result,
    run_recipe,
)


HASH = "a" * 64
COMMIT = "b" * 40


def recipe(**overrides):
    values = {
        "experiment_id": "validation-416-every2",
        "partition": "validation",
        "input_size": 416,
        "frame_skip_interval": 2,
        "frame_limit": 3,
        "source_frame_count": 3,
        "inference_frame_count": 2,
        "dataset_root": "data/research/visual_yolo_v2",
        "package_root": "models/package",
        "training_view_identity_sha256": HASH,
        "membership_sha256": HASH,
        "package_identity_sha256": HASH,
        "model_identity_sha256": HASH,
        "model_file_sha256": HASH,
        "preprocessing_configuration_id": HASH,
        "confidence_threshold": 0.37,
        "runtime_backend": "ultralytics-onnxruntime",
        "device": "cpu",
        "software_commit_sha": COMMIT,
    }
    return ExperimentRecipe(**{**values, **overrides})


class SandboxExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "outputs/sandbox/experiments/validation-416-every2"
        self.output.mkdir(parents=True)
        self.dataset = self.root / "data/research/visual_yolo_v2"
        labels = self.dataset / "labels/validation"
        labels.mkdir(parents=True)
        (labels / "0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (labels / "1.txt").write_text("")
        (labels / "2.txt").write_text("")
        self.rows = [
            {"sample_id": f"sample-{index}",
             "image_relative_path": f"images/validation/{index}.png",
             "label_relative_path": f"labels/validation/{index}.txt"}
            for index in range(3)
        ]
        self.recipe = recipe()
        write_json(self.output / "recipe.json", self.recipe.to_record())

    def tearDown(self):
        self.temporary.cleanup()

    def fake_execution(self, *_args):
        prediction = [{
            "class_name": "transformer", "confidence": 0.9,
            "bbox": [768.0, 432.0, 1152.0, 648.0],
        }]
        raw = [
            {"sample_id": "sample-0", "predictions": prediction, "speed_ms": {}},
            {"sample_id": "sample-2", "predictions": [], "speed_ms": {}},
        ]
        timings = {
            "preprocess_ms": [1.0, 1.0], "inference_ms": [2.0, 2.0],
            "postprocess_ms": [0.5, 0.5], "end_to_end_ms": [3.5, 3.5],
        }
        return raw, {"sample-0": prediction, "sample-2": []}, timings, 0.1

    @patch("src.sandbox.experiment_runner.runtime_environment",
           return_value=({"runtime": "test"}, HASH))
    @patch("src.sandbox.experiment_runner._execute_model")
    @patch("src.sandbox.experiment_runner._load_context")
    def test_run_materializes_identity_bound_aggregate_report(
        self, context, execute, _environment
    ):
        context.return_value = (
            self.recipe, self.output, self.dataset, self.root / "model.onnx", self.rows
        )
        execute.side_effect = self.fake_execution
        result = run_recipe(self.root, self.output / "recipe.json")
        self.assertEqual(result["frame_counts"], {
            "source": 3, "inferred": 2, "skipped": 1, "labelled": 1,
        })
        self.assertEqual(result["metrics"]["precision"], 1.0)
        self.assertEqual(result["metrics"]["recall"], 1.0)
        self.assertEqual(result["metric_scope"], "diagnostic_fixed_threshold_no_ap")
        self.assertEqual(
            inspect_result(self.output / "result.json")["result"], result
        )
        status = json.loads((self.output / "status.json").read_text())
        self.assertEqual(status["state"], "complete")

    @patch("src.sandbox.experiment_runner._execute_model",
           side_effect=RuntimeError("backend failed"))
    @patch("src.sandbox.experiment_runner._load_context")
    def test_failure_preserves_inspectable_status(self, context, _execute):
        context.return_value = (
            self.recipe, self.output, self.dataset, self.root / "model.onnx", self.rows
        )
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            run_recipe(self.root, self.output / "recipe.json")
        status = json.loads((self.output / "status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertIn("backend failed", status["error"])

    def test_recipe_path_must_be_inside_experiment_root(self):
        outside = self.root / "recipe.json"
        outside.write_text("{}")
        with self.assertRaisesRegex(ValueError, "outside"):
            _approved_recipe_path(self.root, outside)

    def test_result_tampering_is_rejected(self):
        result = {"value": 1}
        from src.ml.artifacts import object_sha256
        result["result_identity_sha256"] = object_sha256(result)
        path = self.output / "result.json"
        write_json(path, result)
        loaded = json.loads(path.read_text())
        loaded["value"] = 2
        write_json(path, loaded)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            inspect_result(path)


if __name__ == "__main__":
    unittest.main()
