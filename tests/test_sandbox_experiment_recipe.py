import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.cli import sandbox
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.sandbox.experiment_recipe import (
    ExperimentRecipe,
    inspect_recipe,
    materialize_recipe,
)


HASH = "a" * 64
COMMIT = "b" * 40


def training_identity():
    return TrainingViewIdentity(
        source_development_dataset_identity="1" * 64,
        source_validation_dataset_identity="2" * 64,
        sampling_algorithm="balanced",
        sampling_seed=7,
        train_membership_sha256="3" * 64,
        validation_membership_sha256="4" * 64,
        full_validation_membership_sha256="5" * 64,
        labels_manifest_sha256="6" * 64,
        class_order_identity="7" * 64,
        train_frame_count=1,
        validation_frame_count=2,
        full_validation_frame_count=3,
        train_class_counts={name: 1 for name in ("transformer", "switchgear", "capacitor_bank", "reactor")},
        validation_class_counts={name: 1 for name in ("transformer", "switchgear", "capacitor_bank", "reactor")},
        train_no_target_count=0,
        validation_no_target_count=0,
    )


class SandboxExperimentRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data/research/visual_yolo_v2"
        identity = training_identity()
        write_json(self.dataset / "identity/training_view_identity.json", identity.to_record())
        for partition, count in (("validation", 2), ("full_validation", 3)):
            rows = [
                {
                    "sample_id": f"{partition}-{index}",
                    "image_relative_path": f"images/{partition}/{index}.png",
                    "label_relative_path": f"labels/{partition}/{index}.txt",
                }
                for index in range(count)
            ]
            path = self.dataset / "identity" / f"{partition}_membership.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            digest = file_sha256(path)
            field = f"{partition}_membership_sha256"
            values = identity.identity_record()
            values[field] = digest
            identity = TrainingViewIdentity(**values)
        write_json(self.dataset / "identity/training_view_identity.json", identity.to_record())
        self.package = self.root / "models/equipment/visual-yolo11n-baseline-v2-package"
        self.package.mkdir(parents=True)
        model = {
            "model_identity_sha256": "8" * 64,
            "preprocessing_configuration_id": "9" * 64,
            "runtime_backend": "ultralytics-onnxruntime",
        }
        write_json(self.package / "model_identities.json", {str(size): model for size in (320, 416, 640)})
        self.package_record = {
            "training_view_identity_sha256": identity.training_view_identity_sha256,
            "package_identity_sha256": "a" * 64,
            "frozen_confidence_threshold": 0.37,
            "exports": {str(size): {"sha256": chr(98 + index) * 64} for index, size in enumerate((320, 416, 640))},
        }

    def tearDown(self):
        self.temporary.cleanup()

    @patch("src.sandbox.experiment_recipe.ModelIdentity.from_record")
    @patch("src.sandbox.experiment_recipe.validate_yolo_package")
    @patch("src.sandbox.experiment_recipe.git_commit", return_value=COMMIT)
    def test_materializes_bounded_non_blind_recipe(self, _commit, package, model):
        package.return_value = self.package_record
        model.return_value.model_identity_sha256 = "8" * 64
        model.return_value.preprocessing_configuration_id = "9" * 64
        model.return_value.runtime_backend = "ultralytics-onnxruntime"
        recipe, output = materialize_recipe(
            self.root, "validation-416-every2", partition="validation",
            input_size=416, frame_skip_interval=2, frame_limit=2,
        )
        self.assertEqual(recipe.inference_frame_count, 1)
        self.assertEqual(recipe.confidence_threshold, 0.37)
        self.assertEqual(recipe.software_commit_sha, COMMIT)
        self.assertEqual(inspect_recipe(output / "recipe.json")["recipe"], recipe.to_record())

    def test_recipe_rejects_blind_partition_and_tampering(self):
        values = {
            "experiment_id": "valid-recipe", "partition": "blind",
            "input_size": 416, "frame_skip_interval": 1, "frame_limit": 1,
            "source_frame_count": 1, "inference_frame_count": 1,
            "selection_algorithm": "uniform_partition_bins_v1",
            "dataset_root": "dataset", "package_root": "package",
            "training_view_identity_sha256": HASH, "membership_sha256": HASH,
            "package_identity_sha256": HASH, "model_identity_sha256": HASH,
            "model_file_sha256": HASH, "preprocessing_configuration_id": HASH,
            "confidence_threshold": 0.37, "runtime_backend": "onnx",
            "device": "cpu", "software_commit_sha": COMMIT,
        }
        with self.assertRaisesRegex(ValueError, "non-blind"):
            ExperimentRecipe(**values)
        values["partition"] = "validation"
        recipe = ExperimentRecipe(**values).to_record()
        recipe["input_size"] = 640
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ExperimentRecipe.from_record(recipe)

    def test_cli_parser_exposes_bounded_recipe_choices(self):
        args = sandbox.build_parser().parse_args([
            "recipe-create", "--name", "validation-320-every3",
            "--partition", "validation", "--input-size", "320",
            "--frame-skip", "3", "--frame-limit", "256",
        ])
        self.assertEqual(args.input_size, 320)
        self.assertEqual(args.frame_skip, 3)
        self.assertEqual(args.frame_limit, 256)

    @patch("src.cli.sandbox.materialize_recipe")
    def test_cli_recipe_create_reports_materialized_identity(self, materialize):
        recipe = Mock()
        recipe.to_record.return_value = {"recipe_identity_sha256": HASH}
        output = self.root / "outputs/sandbox/experiments/example"
        materialize.return_value = (recipe, output)
        self.assertEqual(sandbox.main(["--project-root", str(self.root),
                                      "recipe-create", "--name", "example"]), 0)
        materialize.assert_called_once()

    @patch("src.sandbox.experiment_recipe.ModelIdentity.from_record")
    @patch("src.sandbox.experiment_recipe.validate_yolo_package")
    @patch("src.sandbox.experiment_recipe.git_commit", return_value="dirty-dirty")
    def test_dirty_commit_is_rejected(self, _commit, package, model):
        package.return_value = self.package_record
        model.return_value.model_identity_sha256 = "8" * 64
        model.return_value.preprocessing_configuration_id = "9" * 64
        model.return_value.runtime_backend = "ultralytics-onnxruntime"
        with self.assertRaisesRegex(ValueError, "clean tracked commit"):
            materialize_recipe(self.root, "dirty-recipe")


if __name__ == "__main__":
    unittest.main()
