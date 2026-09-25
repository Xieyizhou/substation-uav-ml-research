import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/research/ml_training_recovery_v1"


class VisualBridgeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admission = json.loads((BASE / "visual-bridge-supplement-v2/development-admission.json").read_text())
        cls.out = BASE / "visual-bridge-training-v1"
        cls.protocol = json.loads((cls.out / "protocol.json").read_text())
        cls.progress = json.loads((cls.out / "training-progress.json").read_text())
        cls.evaluation = json.loads((cls.out / "development-evaluation.json").read_text())
        cls.completion = json.loads((cls.out / "completion.json").read_text())

    def test_admission_is_scoped_and_protected_clean(self):
        self.assertEqual(self.admission["status"], "eligible_for_frozen_development_training")
        self.assertEqual(self.admission["frame_count"], 72)
        self.assertEqual(self.admission["subset_counts"], {"bridge_positive": 48, "hard_negative": 24})
        self.assertEqual(self.admission["protected_overlap"], {"file_sha256": 0, "pixel_sha256": 0, "dhash_distance_le_2": 0})
        self.assertTrue(all(row["development_training_eligible"] for row in self.admission["entries"]))
        self.assertTrue(all(not row["training_admitted"] and not row["promotable"] for row in self.admission["entries"]))

    def test_arm_membership_and_controls_are_frozen(self):
        self.assertEqual(self.protocol["status"], "frozen_before_training")
        self.assertEqual({arm: len(value["member_ids"]) for arm, value in self.protocol["membership"].items()},
                         {"A": 114, "B": 162, "C": 138, "D": 186})
        controls = self.protocol["controls"]
        self.assertEqual((controls["input_size"], controls["optimizer"], controls["epochs"], controls["batch"], controls["optimizer_steps"]),
                         (640, "AdamW", 10, 6, 100))
        self.assertFalse(self.protocol["training_admitted"])
        self.assertFalse(self.protocol["promotable"])
        self.assertEqual(self.protocol["unseen_scene_status"], "sealed_not_evaluated")

    def test_lineage_groups_are_never_split_within_an_arm(self):
        rows = {row["member_id"]: row for row in self.protocol["pool_rows"]}
        candidate_groups = {}
        for row in self.protocol["pool_rows"]:
            if row.get("pair_id"):
                candidate_groups.setdefault(row["pair_id"], set()).add(row["member_id"])
        self.assertEqual(len(candidate_groups), 28)
        self.assertEqual({size: sum(len(value) == size for value in candidate_groups.values()) for size in (2, 3)}, {2: 12, 3: 16})
        for arm, spec in self.protocol["membership"].items():
            members = set(spec["member_ids"])
            for group in candidate_groups.values():
                self.assertIn(len(group & members), (0, len(group)), arm)
            self.assertEqual(len(members), len(spec["member_ids"]))
            self.assertTrue(all(Path(rows[item]["image_path"]).is_file() for item in members))

    def test_schedules_have_fixed_draws_and_exposure_records(self):
        for arm in "ABCD":
            for seed in (7, 17, 27):
                schedule = self.protocol["schedules"][arm][str(seed)]
                self.assertEqual((len(schedule), {len(epoch) for epoch in schedule}), (10, {60}))
                self.assertEqual(self.protocol["exposures"][arm][str(seed)]["draws"], 600)

    def test_all_training_cells_completed_with_frozen_exposure(self):
        self.assertEqual(self.progress["status"], "complete_pending_evaluation")
        self.assertEqual(len(self.progress["completed_cells"]), 12)
        self.assertTrue(all(row["exposure_verified"] for row in self.progress["completed_cells"]))
        for row in self.progress["completed_cells"]:
            receipt = json.loads((self.out / f"arm-{row['arm']}-seed-{row['seed']}/completion.json").read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual((receipt["observed_draws"], receipt["optimizer_steps"]), (600, 100))
            self.assertTrue(receipt["exposure_verified"])

    def test_frozen_policy_rejects_all_arms_and_keeps_test_sealed(self):
        self.assertEqual(self.evaluation["status"], "development_complete_no_candidate")
        self.assertIsNone(self.evaluation["selected_arm"])
        self.assertTrue(all(not value["passed"] for value in self.evaluation["policy_results"].values()))
        expected = {
            "A": (0.9166666666666666, 0.08333333333333333, 0.4166666666666667),
            "B": (0.8888888888888888, 0.4722222222222222, 0.4097222222222222),
            "C": (0.9722222222222222, 0.08333333333333333, 0.2222222222222222),
            "D": (0.9166666666666666, 0.4722222222222222, 0.3680555555555556),
        }
        for arm, values in expected.items():
            actual = (
                self.evaluation["aggregate"][arm]["original"]["planned_instance_hit_rate"]["mean"],
                self.evaluation["aggregate"][arm]["material"]["planned_instance_hit_rate"]["mean"],
                self.evaluation["aggregate"][arm]["no_target"]["frame_false_positive_rate"]["mean"],
            )
            for observed, wanted in zip(actual, values):
                self.assertAlmostEqual(observed, wanted)
        self.assertEqual(self.evaluation["unseen_scene_status"], "sealed_not_evaluated")
        self.assertFalse(self.evaluation["training_admitted"])
        self.assertFalse(self.evaluation["promotable"])

    def test_completion_binds_grid_and_rejects_promotion(self):
        self.assertEqual(self.completion["status"], "development_complete_no_candidate")
        self.assertFalse(self.completion["candidate_selected"])
        self.assertIsNone(self.completion["selected_arm"])
        self.assertEqual(len(self.completion["training_cells"]), 12)
        self.assertEqual(self.completion["protocol_identity"], self.protocol["identity"])
        self.assertEqual(self.completion["training_progress_identity"], self.progress["identity"])
        self.assertEqual(self.completion["development_evaluation_identity"], self.evaluation["identity"])
        self.assertEqual(self.completion["unseen_scene_status"], "sealed_not_evaluated")
        self.assertFalse(self.completion["protected_label_accessed"])
        self.assertFalse(self.completion["training_admitted"])
        self.assertFalse(self.completion["promotable"])


if __name__ == "__main__":
    unittest.main()
