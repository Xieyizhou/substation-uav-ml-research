import unittest

from scripts.vision.audit_expected_class_presence import classify


class ExpectedClassPresenceTests(unittest.TestCase):
    def test_expected_target_present_is_observable(self):
        row = {
            "view_id": "v1",
            "map_id": "m1",
            "expected_category": "switchgear",
            "truth": {"objects": [{"class_name": "switchgear"}]},
        }
        result = classify(row)
        self.assertTrue(result["expected_class_present"])
        self.assertEqual(result["status"], "observable_expected_target")

    def test_expected_target_absence_is_a_hold(self):
        row = {
            "view_id": "v2",
            "map_id": "m1",
            "expected_category": "switchgear",
            "truth": {"objects": [{"class_name": "transformer"}]},
        }
        result = classify(row)
        self.assertFalse(result["expected_class_present"])
        self.assertEqual(result["status"], "expected_target_absent")

    def test_non_target_plan_is_not_recall_data(self):
        row = {
            "view_id": "v3",
            "map_id": "m1",
            "expected_category": "cabinet",
            "truth": {"objects": []},
        }
        result = classify(row)
        self.assertIsNone(result["expected_class_present"])
        self.assertEqual(result["status"], "non_target_or_background_plan")
