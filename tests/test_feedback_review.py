from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.inspection.config import InspectionConfig
from src.sandbox.feedback_review import FeedbackReviewStore, validate_decision
from src.sandbox.feedback_dataset import register_feedback_dataset
from src.sandbox.workbench_datasets import import_yolo_dataset
from src.sandbox.workbench_membership import verify_reviewed_dataset
from src.sandbox.workbench_models import WorkbenchDataset
from src.sandbox.workbench_runner import _prepare_view
from src.vision.collection.feedback_recorder import FeedbackRecorder
from tests.test_feedback_recorder import event


def decision(**changes):
    return dict(decision="accepted", boxes=[dict(class_name="switchgear", xyxy=[2, 3, 12, 18])],
                no_target=False, complete_annotation=True, reviewer="test reviewer", note="corrected by review", **changes)


class FeedbackReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = InspectionConfig.defaults(self.root)
        self.collection = self.root / "outputs/sandbox/feedback/test"
        with_recorder = FeedbackRecorder(self.collection, {"recording": "source-a"})
        with_recorder.record(event(200), [dict(class_name="reactor", confidence=0.99, bbox=[0, 0, 30, 30])])
        with_recorder.close()
        self.store = FeedbackReviewStore(self.config)
        self.collection_id = self.store.collections()[0]["collection_id"]
        self.sample_id = self.store.detail(self.collection_id)["samples"][0]["sample_id"]
        self.original = (self.collection / "samples" / (self.sample_id + ".json")).read_bytes()

    def base(self):
        root = self.root / "base"
        for split, offset in (("train", 1), ("val", 20)):
            for name in ("images", "labels"):
                (root / name / split).mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (32, 32), (offset+index, 15, 25)).save(root / "images" / split / f"{index}.png")
                (root / "labels" / split / f"{index}.txt").write_text(f"{index} 0.5 0.5 0.4 0.4\n")
        (root / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [transformer, switchgear, capacitor_bank, reactor]\n")
        return import_yolo_dataset(root, self.config.workbench_datasets_root, "base")

    def register(self):
        return register_feedback_dataset(self.config, "increment", "base", [self.collection_id])

    def test_predictions_never_become_labels_without_explicit_review(self):
        self.base()
        self.assertIsNone(self.store.sample_detail(self.collection_id, self.sample_id)["review"])
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            self.register()
        self.assertFalse((self.config.workbench_datasets_root / "increment").exists())
        self.store.save(self.collection_id, self.sample_id, decision())
        dataset = self.register()
        root = Path(dataset["dataset_root"])
        members = json.loads((root / "membership.json").read_text())
        added = [row for row in members if row["annotation_source"] == "explicit_review"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["split"], "train")
        self.assertTrue((root / added[0]["label_path"]).read_text().startswith("1 "))
        self.assertEqual(dataset["split_counts"], {"train": 5, "validation": 4})
        self.assertEqual((self.collection / "samples" / (self.sample_id + ".json")).read_bytes(), self.original)

    def test_negative_requires_explicit_no_target_and_full_image_review(self):
        value = decision(); value["boxes"] = []
        with self.assertRaisesRegex(ValueError, "explicitly"):
            self.store.save(self.collection_id, self.sample_id, value)
        value["no_target"] = True; value["complete_annotation"] = False
        with self.assertRaisesRegex(ValueError, "complete"):
            self.store.save(self.collection_id, self.sample_id, value)
        value["complete_annotation"] = True
        saved = self.store.save(self.collection_id, self.sample_id, value)
        self.assertTrue(saved["no_target"])

    def test_concurrent_review_updates_require_the_current_identity(self):
        first = self.store.save(self.collection_id, self.sample_id, decision())
        with self.assertRaisesRegex(ValueError, "another window"):
            self.store.save(self.collection_id, self.sample_id, decision())
        next_value = decision(); next_value["decision"] = "rejected"
        second = self.store.save(self.collection_id, self.sample_id, next_value, first["review_identity_sha256"])
        self.assertEqual(second["previous_review_identity"], first["review_identity_sha256"])
        self.assertTrue((self.store.reviews / self.collection_id / self.sample_id / (first["review_identity_sha256"] + ".json")).is_file())

    def test_invalid_boxes_paths_and_changed_pixels_rejected(self):
        for box in ([0, 0, float("nan"), 10], [0, 0, 100, 100], [1, 1, 1, 2]):
            value = decision(); value["boxes"][0]["xyxy"] = box
            with self.assertRaises(ValueError):
                validate_decision(value, dict(width=32, height=32))
        with self.assertRaises(ValueError):
            self.store.resolve("../escape")
        next((self.collection / "frames").glob("*.png")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image changed"):
            self.store.save(self.collection_id, self.sample_id, decision())

    def test_rejected_collection_is_not_registered(self):
        self.base(); value = decision(); value["decision"] = "rejected"
        self.store.save(self.collection_id, self.sample_id, value)
        with self.assertRaisesRegex(ValueError, "explicitly accepted"):
            self.register()

    def test_registered_snapshot_survives_new_review_and_rejects_label_tamper(self):
        self.base(); first = self.store.save(self.collection_id, self.sample_id, decision())
        record = self.register(); dataset = WorkbenchDataset.from_record(record)
        value = decision(); value["boxes"][0]["class_name"] = "transformer"
        self.store.save(self.collection_id, self.sample_id, value, first["review_identity_sha256"])
        members = verify_reviewed_dataset(dataset)
        added = next(row for row in members if row["annotation_source"] == "explicit_review")
        self.assertEqual(added["review_identity_sha256"], first["review_identity_sha256"])
        (Path(dataset.dataset_root) / added["label_path"]).write_text("")
        with self.assertRaisesRegex(ValueError, "label changed"):
            _prepare_view(dataset, self.root / "run", dict(train_limit=256, validation_limit=64))

    def test_training_budget_keeps_every_accepted_feedback_sample(self):
        self.base(); self.store.save(self.collection_id, self.sample_id, decision())
        dataset = WorkbenchDataset.from_record(self.register())
        view, summary = _prepare_view(dataset, self.root / "run", dict(train_limit=3, validation_limit=64))
        members = json.loads((view / "membership.json").read_text())
        self.assertEqual(len([m for m in members if m["split"] == "train"]), 3)
        self.assertIn("train-000008", [m["sample_id"] for m in members])

    def test_missing_or_extra_sample_invalidates_completed_membership(self):
        (self.collection / "samples" / (self.sample_id + ".json")).unlink()
        with self.assertRaisesRegex(ValueError, "membership changed"):
            self.store.detail(self.collection_id)


if __name__ == "__main__":
    unittest.main()
