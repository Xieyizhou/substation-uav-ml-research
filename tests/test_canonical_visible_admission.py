import copy
import unittest

from src.vision.canonical.visible_admission import adapt_visible_truth, review_frame


def truth(bounds=None):
    return {"validation_status": "valid", "image_width": 1920,
            "image_height": 1080, "objects": [{"annotation_id": "a",
            "class_name": "transformer", "class_id": 0,
            "bbox_xyxy": bounds or [0, 239, 595, 1079],
            "truncation_status": "not_truncated"}]}


class VisibleAdmissionTests(unittest.TestCase):
    def test_conversion_preserves_source_and_identity(self):
        raw = truth()
        original = copy.deepcopy(raw)
        result = adapt_visible_truth(raw, annotation_mode="visible_2d")
        self.assertEqual(raw, original)
        self.assertEqual(result["objects"][0]["bbox_xyxy"], [0, 239, 596, 1080])
        self.assertEqual(result["objects"][0]["image_edge_contact"], ["left", "bottom"])
        self.assertEqual(result["objects"][0]["truncation_status"], "unknown")
        self.assertEqual(result, adapt_visible_truth(raw, annotation_mode="visible_2d"))

    def test_single_pixel_is_not_silently_deleted(self):
        result = adapt_visible_truth(truth([5, 5, 5, 5]), annotation_mode="visible_2d")
        self.assertEqual(result["objects"][0]["bbox_xyxy"], [5, 5, 6, 6])

    def test_reject_legacy_and_invalid_bounds(self):
        with self.assertRaises(ValueError):
            adapt_visible_truth(truth(), annotation_mode="full_2d")
        for bounds in ([0, 0, 1920, 20], [0, 0, float("nan"), 20],
                       [0, 0, 20.5, 20], [10, 0, 9, 20]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                adapt_visible_truth(truth(bounds), annotation_mode="visible_2d")

    def test_missing_review_quarantines_whole_frame(self):
        result = adapt_visible_truth(truth(), annotation_mode="visible_2d")
        review = review_frame(result, {}, image_sha256="image")
        self.assertEqual(review["status"], "quarantined")
        self.assertEqual(review["retained_annotation_count"], 1)
        self.assertFalse(review["drop_annotations"])

    def test_review_pass_is_not_training_admission(self):
        result = adapt_visible_truth(truth(), annotation_mode="visible_2d")
        review = {"image_sha256": "image", "adapter_identity": result["adapter_identity"],
                  "all_visible_targets_correct": True,
                  "objects": [{"annotation_id": "a", "status": "accepted",
                               "visible_extent_confirmed": True,
                               "class_evidence_confirmed": True,
                               "framing_status": "accepted"}]}
        status = review_frame(result, review, image_sha256="image")
        self.assertEqual(status["status"], "semantic_review_passed")
        self.assertFalse(status["training_admitted"])
        review["objects"][0]["status"] = "ignored"
        self.assertEqual(review_frame(result, review, image_sha256="image")["status"], "quarantined")

    def test_edge_contact_requires_evidence_not_automatic_rejection(self):
        result = adapt_visible_truth(truth(), annotation_mode="visible_2d")
        review = {"image_sha256": "image", "adapter_identity": result["adapter_identity"],
                  "all_visible_targets_correct": True,
                  "objects": [{"annotation_id": "a", "status": "accepted"}]}
        status = review_frame(result, review, image_sha256="image")
        self.assertIn("framing_unconfirmed_or_rejected", status["reasons"])
        review["objects"][0].update(visible_extent_confirmed=True,
                                   class_evidence_confirmed=True,
                                   framing_status="accepted")
        self.assertEqual(review_frame(result, review, image_sha256="image")["status"],
                         "semantic_review_passed")
        review["objects"][0]["framing_status"] = "rejected"
        self.assertEqual(review_frame(result, review, image_sha256="image")["status"],
                         "quarantined")

    def test_empty_truth_needs_background_confirmation(self):
        raw = truth()
        raw["objects"] = []
        result = adapt_visible_truth(raw, annotation_mode="visible_2d")
        review = {"image_sha256": "image", "adapter_identity": result["adapter_identity"],
                  "all_visible_targets_correct": True, "objects": []}
        self.assertIn("background_unconfirmed", review_frame(result, review, image_sha256="image")["reasons"])


if __name__ == "__main__":
    unittest.main()
