import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.ml.artifacts import file_sha256
from src.sandbox.workbench_datasets import (
    import_yolo_dataset,
    list_workbench_datasets,
)
from src.sandbox.workbench_models import WorkbenchExperimentRecipe


class WorkbenchDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "managed"
        (self.source / "images/train").mkdir(parents=True)
        (self.source / "images/val").mkdir(parents=True)
        (self.source / "labels/train").mkdir(parents=True)
        (self.source / "labels/val").mkdir(parents=True)
        (self.source / "dataset.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\n"
            "names:\n  0: transformer\n  1: switchgear\n"
            "  2: capacitor_bank\n  3: reactor\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def sample(self, split, name, label):
        image = self.source / "images" / split / f"{name}.png"
        Image.new("RGB", (16, 16), (hash(name) % 255, 12, 24)).save(image)
        (self.source / "labels" / split / f"{name}.txt").write_text(
            label, encoding="utf-8"
        )
        return image

    def test_imports_immutable_yolo_dataset_and_empty_labels(self):
        source = self.sample("train", "positive", "0 0.5 0.5 0.5 0.5\n")
        self.sample("val", "negative", "")
        result = import_yolo_dataset(self.source, self.output, "external-1")
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1})
        self.assertEqual(result["no_target_counts"]["validation"], 1)
        managed = self.output / "external-1/images/train/train-000000.png"
        before = file_sha256(managed)
        Image.new("RGB", (16, 16), "white").save(source)
        self.assertEqual(file_sha256(managed), before)
        rows = list_workbench_datasets(self.root, self.output)
        self.assertEqual(rows[0]["dataset_id"], "external-1")

    def test_rejects_missing_label_and_cleans_partial_import(self):
        Image.new("RGB", (16, 16), "red").save(
            self.source / "images/train/missing.png"
        )
        self.sample("val", "valid", "")
        with self.assertRaises(FileNotFoundError):
            import_yolo_dataset(self.source, self.output, "broken")
        self.assertFalse((self.output / "broken").exists())

    def test_rejects_unmapped_class_and_out_of_bounds_box(self):
        self.sample("train", "unknown", "4 0.5 0.5 0.2 0.2\n")
        self.sample("val", "valid", "")
        with self.assertRaisesRegex(ValueError, "unmapped class"):
            import_yolo_dataset(self.source, self.output, "unknown")
        (self.source / "labels/train/unknown.txt").write_text(
            "0 0.1 0.1 0.4 0.4\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            import_yolo_dataset(self.source, self.output, "outside")

    def test_discovers_native_training_identity_without_rewriting_it(self):
        identity = self.root / "data/research/visual_yolo_native/identity"
        identity.mkdir(parents=True)
        value = {
            "training_view_identity_sha256": "a" * 64,
            "train_frame_count": 10, "validation_frame_count": 4,
            "train_class_counts": {}, "validation_class_counts": {},
            "train_no_target_count": 2, "validation_no_target_count": 1,
        }
        path = identity / "training_view_identity.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        before = path.read_bytes()
        rows = list_workbench_datasets(self.root, self.output)
        self.assertEqual(rows[0]["source_type"], "native_training_view")
        self.assertEqual(path.read_bytes(), before)


class WorkbenchRecipeTests(unittest.TestCase):
    def test_preset_override_and_identity_round_trip(self):
        recipe = WorkbenchExperimentRecipe.create(
            experiment_id="run-1", dataset_id="dataset",
            dataset_identity_sha256="a" * 64, preset="quick",
            pretrained_weights_sha256="b" * 64,
            overrides={"epochs": 5, "device": "cpu"},
        )
        record = recipe.to_record()
        self.assertEqual(record["parameters"]["epochs"], 5)
        self.assertEqual(record["parameters"]["imgsz"], 416)
        self.assertEqual(
            WorkbenchExperimentRecipe.from_record(record).recipe_identity_sha256,
            recipe.recipe_identity_sha256,
        )

    def test_rejects_non_allowlisted_parameter_and_tamper(self):
        with self.assertRaisesRegex(ValueError, "unsupported workbench parameter"):
            WorkbenchExperimentRecipe.create(
                experiment_id="run", dataset_id="data",
                dataset_identity_sha256="a" * 64, preset="smoke",
                pretrained_weights_sha256="b" * 64,
                overrides={"epochs": 200},
            )
        recipe = WorkbenchExperimentRecipe.create(
            experiment_id="run", dataset_id="data",
            dataset_identity_sha256="a" * 64, preset="smoke",
            pretrained_weights_sha256="b" * 64,
        ).to_record()
        recipe["parameters"]["batch"] = 16
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            WorkbenchExperimentRecipe.from_record(recipe)


if __name__ == "__main__":
    unittest.main()
