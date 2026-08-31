import unittest

from src.vision.evaluation.proposal_audit import proposal_metrics


class ProposalAuditTests(unittest.TestCase):
    def test_class_agnostic_match_ignores_predicted_class(self):
        frames = [{
            "sample_id": "one",
            "truth": [{"class_name": "capacitor_bank", "bbox": [0, 0, 10, 10]}],
            "predictions": [{
                "class_name": "transformer", "confidence": 0.9,
                "bbox": [0, 0, 10, 10],
            }],
        }]
        result = proposal_metrics(frames, 0.05)
        self.assertEqual(result["overall_recall"], 1.0)
        self.assertEqual(result["per_class"]["capacitor_bank"]["recall"], 1.0)

    def test_cross_class_duplicates_are_suppressed(self):
        frames = [{
            "sample_id": "one", "truth": [],
            "predictions": [
                {"class_name": "transformer", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                {"class_name": "switchgear", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
            ],
        }]
        result = proposal_metrics(frames, 0.05)
        self.assertEqual(result["proposal_count_max"], 1)
        self.assertEqual(result["false_positive_count"], 1)


if __name__ == "__main__":
    unittest.main()
