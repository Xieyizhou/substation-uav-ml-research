import unittest
from pathlib import Path
import tempfile

from src.vision.evaluation.detection_metrics import (
    box_iou,
    match_detections,
    onnx_equivalence,
    select_confidence_threshold,
    threshold_metrics,
)
from src.vision.evaluation.onnx_gate import _calibration_dataset, calibration_members


def truth(class_name="transformer", *, small=False):
    return {
        "class_name": class_name,
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "small": small,
    }


def prediction(class_name="transformer", confidence=0.9, bbox=None):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox or [0.0, 0.0, 10.0, 10.0],
    }


class DetectionMetricTests(unittest.TestCase):
    def test_iou_and_class_aware_matching(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        matches, false_positives, false_negatives = match_detections(
            [truth()],
            [prediction(), prediction("switchgear")],
            threshold=0.25,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(len(false_negatives), 0)

    def test_threshold_metrics_keep_no_target_and_small_recall_separate(self):
        frames = [
            {
                "truth": [truth(small=True)],
                "predictions": [prediction(confidence=0.8)],
            },
            {
                "truth": [],
                "predictions": [prediction(confidence=0.4)],
            },
        ]
        metrics = threshold_metrics(frames, 0.5)
        self.assertEqual(metrics["small_object_recall"], 1.0)
        self.assertEqual(metrics["no_target_false_positive_rate"], 0.0)

    def test_threshold_selection_uses_macro_f1_and_stable_lower_tie(self):
        frames = [
            {
                "truth": [truth(name) for name in (
                    "transformer",
                    "switchgear",
                    "capacitor_bank",
                    "reactor",
                )],
                "predictions": [
                    prediction(name, 0.6)
                    for name in (
                        "transformer",
                        "switchgear",
                        "capacitor_bank",
                        "reactor",
                    )
                ],
            }
        ]
        selected = select_confidence_threshold(frames)["selected"]
        self.assertEqual(selected["threshold"], 0.05)
        self.assertEqual(selected["macro_f1"], 1.0)

    def test_onnx_equivalence_rejects_unmatched_outputs(self):
        pt = [{"sample_id": "a", "predictions": [prediction()]}]
        same = [{"sample_id": "a", "predictions": [prediction(confidence=0.895)]}]
        self.assertTrue(onnx_equivalence(pt, same)["passed"])
        missing = [{"sample_id": "a", "predictions": []}]
        self.assertFalse(onnx_equivalence(pt, missing)["passed"])

    def test_calibration_is_stratified_and_stable(self):
        rows = []
        for group in (
            "transformer",
            "switchgear",
            "capacitor_bank",
            "reactor",
            None,
        ):
            for index in range(50):
                rows.append(
                    {
                        "sample_id": f"{group}-{index}",
                        "sequence_number": index,
                        "classes": [group] if group else [],
                    }
                )
        first = calibration_members(rows)
        second = calibration_members(list(reversed(rows)))
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual(len(first), 200)

    def test_calibration_dataset_is_valid_for_ultralytics_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "source"
            image = dataset / "images/validation/sample.png"
            label = dataset / "labels/validation/sample.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            label.write_text("", encoding="utf-8")
            row = {
                "image_relative_path": "images/validation/sample.png",
                "label_relative_path": "labels/validation/sample.txt",
            }
            yaml = _calibration_dataset(dataset, root / "calibration", [row])
            content = yaml.read_text(encoding="utf-8")
            self.assertIn("train: images/calibration\n", content)
            self.assertIn("val: images/calibration\n", content)


if __name__ == "__main__":
    unittest.main()
