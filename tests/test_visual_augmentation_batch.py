import json
from collections import Counter
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-240-v1"
FROZEN=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-240-v2/frozen-intake-ledger.json"
TRUSTED=ROOT/"data/research/ml_training_recovery_v1/trusted-training-base-v1/frozen-ledger.json"
TRAINING=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-training-v2/protocol.json"
EVALUATION=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-training-v2/development-evaluation.json"
COMPLETION=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-training-v2/completion.json"
COMPOSITION=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-composition-v1/protocol.json"
COMPOSITION_EVALUATION=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-composition-v1/development-evaluation.json"
COMPOSITION_COMPLETION=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-composition-v1/completion.json"
FAILURE_ANALYSIS_PROTOCOL=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-failure-analysis-v1/protocol.json"
FAILURE_ANALYSIS=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-failure-analysis-v1/analysis.json"


class VisualAugmentationBatchTests(unittest.TestCase):
    def test_matrix_counts_and_lineage_groups_are_frozen(self):
        matrix=json.loads((BASE/"matrix.json").read_text());ledger=json.loads((BASE/"intake-ledger.json").read_text())
        self.assertEqual(matrix["counts"],{"appearance_lighting_positive":144,"hard_negative":48,"regular_positive":48,"total":240})
        self.assertEqual(len(ledger["entries"]),240)
        groups=Counter(row["derivation_group"] for row in ledger["entries"])
        self.assertEqual(sorted(groups.values()).count(6),24)
        self.assertEqual(sorted(groups.values()).count(2),24)
        self.assertEqual(sorted(groups.values()).count(1),48)
        self.assertTrue(all(row["training_admitted"] is False for row in [matrix,ledger]))

    def test_fixed_regression_is_never_a_training_role(self):
        roles=json.loads((BASE/"data-roles.json").read_text())
        fixed=roles["roles"]["fixed_development_regression"]
        self.assertIn("never train",fixed["rule"])
        self.assertIn("not blind",fixed["rule"])

    def test_unseen_scene_is_sealed(self):
        seal=json.loads((BASE/"unseen-scene-test-plan.json").read_text())
        self.assertEqual(seal["status"],"sealed_not_evaluated")
        self.assertEqual(seal["map_layout_id"],"extreme-canonical-v1")
        self.assertEqual(seal["positive_frames"]+seal["planned_negative_frames"],24)
        self.assertTrue(seal["instance_collision_resolutions"])

    def test_review_cannot_finalize_with_implicit_acceptance(self):
        from scripts.vision.finalize_visual_augmentation_positives import finalize
        manifest=json.loads((BASE/"positive-review-v1/manifest.json").read_text())
        with self.assertRaisesRegex(ValueError,"decisions are incomplete"):
            finalize(manifest,set())

    def test_v2_frozen_ledger_has_complete_reviewed_membership(self):
        ledger=json.loads(FROZEN.read_text())
        self.assertEqual(ledger["status"],"frozen")
        self.assertEqual(ledger["counts"],{
            "appearance_lighting_positive":144,"hard_negative":48,
            "regular_positive":48,"total":240})
        self.assertEqual(ledger["independent_pose_groups"],{
            "appearance_lighting_positive":24,"regular_positive":48,
            "hard_negative":24})
        self.assertEqual(len({row["image_sha256"] for row in ledger["entries"]}),240)
        self.assertEqual(len({row["pixel_sha256"] for row in ledger["entries"]}),240)
        self.assertTrue(all(row["review_decision"]=="accepted" for row in ledger["entries"]))
        self.assertFalse(ledger["training_admitted"])
        self.assertFalse(ledger["promotable"])
        self.assertEqual(set(ledger["deduplication"].values())-{True},{0})

    def test_common_base_requires_explicit_hash_bound_review(self):
        ledger=json.loads(TRUSTED.read_text())
        self.assertEqual((ledger["status"],ledger["frame_count"]),("frozen",66))
        self.assertTrue(all(row["review_decision"]=="accepted" for row in ledger["entries"]))
        self.assertTrue(all(row["review_nature"]=="AI-assisted" for row in ledger["entries"]))
        self.assertTrue(all(row["training_image_sha256"] and row["training_label_sha256"] for row in ledger["entries"]))

    def test_abcd_membership_and_schedules_are_complete(self):
        protocol=json.loads(TRAINING.read_text())
        self.assertEqual({arm:len(row["member_ids"]) for arm,row in protocol["membership"].items()},
                         {"A":114,"B":258,"C":162,"D":306})
        self.assertEqual(protocol["controls"]["optimizer_steps"],100)
        for arm,membership in protocol["membership"].items():
            expected=set(membership["member_ids"])
            for seed in ("7","17","27"):
                draws=[member for epoch in protocol["schedules"][arm][seed] for member in epoch]
                self.assertEqual(len(draws),600)
                self.assertEqual(set(draws),expected)
        training_paths={row["image_path"] for row in protocol["pool_rows"]}
        self.assertFalse(any("paired-visual-factors-v1" in path for path in training_paths))
        self.assertFalse(protocol["training_admitted"])
        self.assertFalse(protocol["promotable"])

    def test_training_grid_and_evaluation_close_without_selection(self):
        evaluation=json.loads(EVALUATION.read_text()); completion=json.loads(COMPLETION.read_text())
        self.assertEqual(evaluation["status"],"development_evaluation_complete")
        self.assertEqual(len(evaluation["results"]),13)
        self.assertTrue(all(len(evaluation["results"][f"{arm}_{seed}"]["paired_rows"])==48
                            for arm in "ABCD" for seed in (7,17,27)))
        self.assertTrue(all(len(evaluation["results"][f"{arm}_{seed}"]["negative_rows"])==48
                            for arm in "ABCD" for seed in (7,17,27)))
        self.assertEqual(completion["status"],"complete_with_tradeoff_no_candidate_selected")
        self.assertEqual(len(completion["training_cells"]),12)
        self.assertFalse(completion["candidate_selected"])
        self.assertEqual(completion["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(completion["training_admitted"])
        self.assertFalse(completion["promotable"])

    def test_composition_protocol_is_frozen_before_training(self):
        protocol=json.loads(COMPOSITION.read_text())
        self.assertEqual(protocol["status"],"frozen_before_training")
        self.assertEqual(protocol["unseen_scene_status"],"sealed_not_evaluated")
        self.assertEqual(protocol["quotas"],{
            "E":{"base":180,"regular":150,"appearance":150,"negative":120},
            "F":{"base":162,"regular":132,"appearance":186,"negative":120}})
        expected=set(protocol["membership"]["member_ids"])
        self.assertEqual(len(expected),306)
        for arm in "EF":
            for seed in ("7","17","27"):
                draws=[member for epoch in protocol["schedules"][arm][seed] for member in epoch]
                self.assertEqual(len(draws),600)
                self.assertEqual(set(draws),expected)
                self.assertEqual(protocol["exposures"][arm][seed]["unique_frames"],306)
        metrics={rule["metric"] for rule in protocol["acceptance_policy"]["required_all"]}
        self.assertIn("original.planned_instance_hit_rate.min_seed",metrics)
        self.assertIn("no_target.frame_false_positive_rate.max_seed",metrics)
        self.assertFalse(protocol["training_admitted"])
        self.assertFalse(protocol["promotable"])

    def test_composition_grid_closes_without_opening_unseen_scene(self):
        protocol=json.loads(COMPOSITION.read_text())
        evaluation=json.loads(COMPOSITION_EVALUATION.read_text())
        completion=json.loads(COMPOSITION_COMPLETION.read_text())
        self.assertEqual(evaluation["status"],"development_complete_no_candidate")
        self.assertIsNone(evaluation["candidate_family"])
        self.assertFalse(evaluation["acceptance_decisions"]["E"]["passed"])
        self.assertFalse(evaluation["acceptance_decisions"]["F"]["passed"])
        self.assertEqual(len(evaluation["results"]),6)
        self.assertTrue(all(len(row["paired_rows"])==48 for row in evaluation["results"].values()))
        self.assertTrue(all(len(row["negative_rows"])==48 for row in evaluation["results"].values()))
        for arm in "EF":
            for seed in (7,17,27):
                row=json.loads((COMPOSITION.parent/f"arm-{arm}-seed-{seed}/completion.json").read_text())
                self.assertEqual(row["protocol_identity"],protocol["identity"])
                self.assertEqual((row["optimizer_steps"],row["observed_draws"]),(100,600))
                self.assertTrue(row["exposure_verified"])
        self.assertEqual(completion["status"],"development_complete_no_candidate")
        self.assertIsNone(completion["candidate_family"])
        self.assertEqual(completion["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(completion["training_admitted"])
        self.assertFalse(completion["promotable"])

    def test_failure_analysis_preserves_lineage_and_seal(self):
        protocol=json.loads(FAILURE_ANALYSIS_PROTOCOL.read_text())
        analysis=json.loads(FAILURE_ANALYSIS.read_text())
        self.assertEqual(protocol["status"],"frozen_before_analysis")
        self.assertEqual(analysis["protocol_identity"],protocol["identity"])
        self.assertEqual((analysis["appearance"]["frames"],analysis["appearance"]["independent_pose_groups"]),(144,24))
        lineage=analysis["appearance"]["lineage_summary"]
        self.assertEqual((lineage["groups_with_six_variants"],lineage["groups_with_one_label_hash"]),(24,24))
        self.assertEqual(lineage["maximum_target_bbox_delta_px"],0.0)
        for row in analysis["appearance"]["category_summary"].values():
            self.assertEqual(row["frames"],36)
            self.assertEqual(row["independent_poses"],6)
            self.assertEqual(set(row["by_actual_material"].values()),{12})
            self.assertEqual(set(row["by_lighting"].values()),{18})
        self.assertEqual(analysis["negative"]["coverage_frames"],{
            "control_building":6,"ordinary_cabinet":10,"utility_pole":32})
        self.assertEqual({arm:analysis["negative"]["families"][arm]["worst_fpr_seed"] for arm in "EF"},{"E":17,"F":7})
        self.assertEqual(analysis["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(analysis["training_admitted"])
        self.assertFalse(analysis["promotable"])


if __name__=="__main__":unittest.main()
