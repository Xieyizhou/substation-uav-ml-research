import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/research/ml_training_recovery_v1/visual-bridge-supplement-v1"
BASE_V2=ROOT/"data/research/ml_training_recovery_v1/visual-bridge-supplement-v2"
NEGATIVE_V2=BASE_V2/"negative-redesign-v2"


class VisualBridgeSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix=json.loads((BASE/"matrix.json").read_text())

    def test_positive_matrix_is_paired_and_far(self):
        self.assertEqual(self.matrix["status"],"frozen_pending_pilot")
        self.assertEqual((self.matrix["positive"]["frame_count"],self.matrix["positive"]["independent_pose_groups"]),(48,16))
        groups=[];pair_sets=[]
        for run in self.matrix["positive"]["runs"]:
            plan=json.loads(Path(run["plan_path"]).read_text());views=plan["calibration_views"]
            self.assertEqual(len(views),16)
            self.assertEqual({row["category"] for row in views},{"capacitor_bank","reactor"})
            self.assertGreaterEqual(min(row["distance"] for row in views),12.0)
            groups.append({row["derivation_group"] for row in views});pair_sets.append({row["pair_id"] for row in views})
            self.assertTrue(all(row["training_admitted"] is False and row["promotable"] is False for row in views))
        self.assertEqual(groups[0],groups[1]);self.assertEqual(groups[1],groups[2])
        self.assertEqual(pair_sets[0],pair_sets[1]);self.assertEqual(pair_sets[1],pair_sets[2])

    def test_world_diffs_are_limited_to_declared_factor(self):
        from scripts.vision.prepare_paired_visual_factors import assert_allowed_world_diff
        paths={row["variant"]:Path(row["plan_path"]).parent/"world.sdf" for row in self.matrix["positive"]["runs"]}
        assert_allowed_world_diff(paths["original"],paths["neutral_bridge"],"material")
        assert_allowed_world_diff(paths["original"],paths["background_bridge"],"background")

    def test_pilot_and_negative_redesign_are_bounded(self):
        pilot=self.matrix["positive_pilot"]
        self.assertEqual((pilot["status"],pilot["frame_count"],pilot["independent_pose_groups"]),("frozen_pending_capture",6,2))
        for run in pilot["runs"]:
            plan=json.loads(Path(run["plan_path"]).read_text())
            self.assertEqual(len(plan["calibration_views"]),2)
            self.assertEqual({row["category"] for row in plan["calibration_views"]},{"capacitor_bank","reactor"})
        negative=self.matrix["negative_redesign"]
        self.assertEqual((negative["frame_count"],negative["independent_pose_groups"]),(24,12))
        self.assertEqual(set(negative["family_frame_counts"].values()),{4})
        custom=[row for row in negative["rows"] if row["source_scene"]!="complex-canonical-v1"]
        self.assertTrue(custom)
        self.assertTrue(all(row["capture_status"]=="planned_pending_source_live_validation" for row in custom))
        self.assertTrue(all(row["empty_truth_required"] and row["visual_target_exclusion_review_required"] for row in negative["rows"]))

    def test_no_admission_or_unseen_access(self):
        self.assertEqual(self.matrix["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(self.matrix["training_admitted"])
        self.assertFalse(self.matrix["promotable"])


class CorrectedVisualBridgeSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1=json.loads((BASE/"matrix.json").read_text())
        cls.matrix=json.loads((BASE_V2/"matrix.json").read_text())

    def test_v2_preserves_failed_v1_identity(self):
        self.assertEqual(self.matrix["schema_version"],2)
        self.assertEqual(self.matrix["superseded_identity"],self.v1["identity"])
        self.assertIn("before Gazebo launch",self.matrix["supersession_reason"])
        self.assertFalse((BASE/"pilot-v1/capture-progress.json").exists())

    def test_every_persisted_world_is_full_2d(self):
        from src.vision.canonical.gates import annotation_mode_from_world
        for group in (self.matrix["positive"]["runs"],self.matrix["positive_pilot"]["runs"]):
            for run in group:
                world=Path(run["plan_path"]).parent/"world.sdf"
                plan=json.loads(Path(run["plan_path"]).read_text())
                self.assertEqual(annotation_mode_from_world(world),"full_2d")
                self.assertEqual(plan["annotation_mode"],"full_2d")
                self.assertEqual(run["actual_annotation_mode"],"full_2d")

    def test_v2_retains_scope_and_seal(self):
        self.assertEqual(self.matrix["positive"]["frame_count"],48)
        self.assertEqual(self.matrix["positive_pilot"]["frame_count"],6)
        self.assertEqual(set(self.matrix["negative_redesign"]["family_frame_counts"].values()),{4})
        self.assertEqual(self.matrix["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(self.matrix["training_admitted"])
        self.assertFalse(self.matrix["promotable"])

    def test_pilot_and_remaining_capture_are_explicitly_reviewed(self):
        pilot=json.loads((BASE_V2/"pilot-v1/review-v1/semantic-review.json").read_text())
        remaining=json.loads((BASE_V2/"remaining-positive-v1/review-v1/semantic-review.json").read_text())
        self.assertEqual((pilot["accepted"],pilot["held"]),(6,0))
        self.assertEqual((remaining["accepted"],remaining["held"]),(42,0))
        self.assertEqual(remaining["maximum_target_bbox_delta_px"],0)
        self.assertTrue(all(row["review_nature"]=="AI-assisted" for row in pilot["frames"]+remaining["frames"]))
        self.assertTrue(all(row["decision"]=="accepted" and row["reason"] for row in pilot["frames"]+remaining["frames"]))

    def test_positive_ledger_has_complete_lineage_and_no_admission(self):
        ledger=json.loads((BASE_V2/"frozen-positive-ledger.json").read_text())
        self.assertEqual(ledger["status"],"positive_supplement_frozen_pending_negative_completion")
        self.assertEqual((ledger["frame_count"],ledger["independent_pose_groups"]),(48,16))
        self.assertEqual(ledger["variant_counts"],{"original":16,"neutral_bridge":16,"background_bridge":16})
        self.assertEqual(ledger["category_counts"],{"capacitor_bank":24,"reactor":24})
        pair_counts={}
        for row in ledger["frames"]:
            pair_counts[row["pair_id"]]=pair_counts.get(row["pair_id"],0)+1
            self.assertEqual(row["annotation"],{
                "actual_mode":"full_2d","check_version":"canonical-collection-gates-v2",
                "hierarchy_mode":"top-level-equipment","label_mode":"visual-instance"})
            self.assertTrue(row["instance_checks"]["category_present"])
            self.assertTrue(row["instance_checks"]["planned_instance_present"])
            self.assertEqual(row["review"]["nature"],"AI-assisted")
            self.assertFalse(row["training_admitted"])
            self.assertFalse(row["promotable"])
        self.assertEqual(set(pair_counts.values()),{3})
        self.assertEqual(ledger["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(ledger["training_admitted"])
        self.assertFalse(ledger["promotable"])

    def test_dedup_keeps_only_designed_sibling_similarity(self):
        pilot=json.loads((BASE_V2/"pilot-v1/review-v1/dedup-audit.json").read_text())
        remaining=json.loads((BASE_V2/"remaining-positive-v1/review-v1/dedup-audit.json").read_text())
        self.assertEqual((pilot["status"],remaining["status"]),("passed","passed"))
        self.assertTrue(all(row["status"].startswith("accepted_no_") for row in pilot["decisions"]+remaining["decisions"]))

    def test_canonical_only_negative_replacement_is_complete(self):
        matrix=json.loads((NEGATIVE_V2/"matrix.json").read_text())
        ledger=json.loads((NEGATIVE_V2/"frozen-negative-ledger.json").read_text())
        self.assertEqual(matrix["counts"],{"frames":24,"independent_pose_groups":12,"lights_per_pose":2})
        self.assertFalse(matrix["off_scope_background_package"]["collection_allowed"])
        self.assertFalse(ledger["off_scope_background_package_used"])
        self.assertEqual((ledger["frame_count"],ledger["independent_pose_groups"]),(24,12))
        self.assertEqual(ledger["lighting_counts"],{"light_normal":12,"light_cool_low":12})
        self.assertEqual(set(ledger["family_counts"].values()),{4})
        pairs={}
        for row in ledger["frames"]:
            pairs.setdefault(row["pair_id"],set()).add(row["lighting_id"])
            self.assertEqual(row["truth_object_count"],0)
            self.assertEqual(row["review"]["decision"],"accepted")
            self.assertEqual(row["review"]["nature"],"AI-assisted")
            self.assertEqual(row["annotation"]["actual_mode"],"full_2d")
            self.assertFalse(row["training_admitted"])
            self.assertFalse(row["promotable"])
        self.assertEqual(len(pairs),12)
        self.assertTrue(all(value=={"light_normal","light_cool_low"} for value in pairs.values()))

    def test_supplement_completion_remains_nonadmitted_and_sealed(self):
        completion=json.loads((BASE_V2/"supplement-completion.json").read_text())
        self.assertEqual(completion["total_frames"],72)
        self.assertEqual((completion["positive_frames"],completion["negative_frames"]),(48,24))
        self.assertFalse(completion["off_scope_background_package_used"])
        self.assertEqual(completion["unseen_scene_status"],"sealed_not_evaluated")
        self.assertFalse(completion["training_started"])
        self.assertFalse(completion["training_admitted"])
        self.assertFalse(completion["promotable"])


if __name__=="__main__": unittest.main()
