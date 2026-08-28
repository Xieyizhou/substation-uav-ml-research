import tempfile
import unittest

from src.vision.training.hard_example_curator import Candidate, apply_bbox_policy, curate, hamming_hex


def candidate(frame, split, sha, phash, map_id="simple", target=True):
    objects = ({"annotation_id": frame, "class_name": "transformer"},) if target else ()
    return Candidate("/collection", "identity", frame, map_id, split, 1, frame + ".ppm", 1.0, sha, phash, objects)


class HardExampleCuratorTests(unittest.TestCase):
    def test_validation_bbox_policy_rejects_truncated_truth(self):
        truncated = candidate("edge", "validation", "a", "0" * 16)
        complete = candidate("complete", "validation", "b", "f" * 16)
        truncated = Candidate(**{**truncated.__dict__, "objects": ({"annotation_id": "edge", "class_name": "transformer", "bbox_xyxy": [0, 100, 500, 900]},)})
        complete = Candidate(**{**complete.__dict__, "objects": ({"annotation_id": "complete", "class_name": "transformer", "bbox_xyxy": [100, 100, 900, 900]},)})
        accepted, rejected = apply_bbox_policy([truncated, complete], {"validation": {"minimum_border_margin_px": 2, "maximum_bbox_width_fraction": 0.9, "maximum_bbox_height_fraction": 0.9}})
        self.assertEqual([row.frame_id for row in accepted], ["complete"])
        self.assertEqual(rejected[0]["reason"], "bbox_touches_frame_boundary")

    def test_seed_policy_keeps_only_bound_target_instance(self):
        row = candidate("frame", "development", "a", "0" * 16)
        row = Candidate(**{**row.__dict__, "seed": 6101, "objects": (
            {"annotation_id": "truth-instance-0057-box-0000", "class_name": "transformer", "bbox_xyxy": [100, 100, 900, 900]},
            {"annotation_id": "truth-instance-0127-box-0001", "class_name": "switchgear", "bbox_xyxy": [100, 100, 900, 900]},
        )})
        accepted, rejected = apply_bbox_policy([row], {
            "target_instance_labels_by_seed": {"6101": [57]},
            "development": {"mode": "drop_invalid_objects"},
        })
        self.assertEqual([item["class_name"] for item in accepted[0].objects], ["transformer"])
        self.assertIn("non_target_instance", {item["reason"] for item in rejected})

    def test_hamming_hex(self):
        self.assertEqual(hamming_hex("0000000000000000", "0000000000000003"), 2)

    def test_exact_duplicate_is_rejected(self):
        rows = [candidate("a", "development", "same", "0" * 16), candidate("b", "development", "same", "f" * 16)]
        selected, rejected, _, _, _ = curate(rows, {("simple", "target", "development"): 1}, 0)
        self.assertEqual([row.frame_id for row in selected], ["a"])
        self.assertIn("exact_duplicate", {row["reason"] for row in rejected})

    def test_near_duplicate_cluster_isolated_to_one_split(self):
        rows = [candidate("dev", "development", "a", "0" * 16), candidate("val", "validation", "b", "0" * 15 + "1")]
        quotas = {("simple", "target", "development"): 1, ("simple", "target", "validation"): 1}
        selected, rejected, _, _, shortfall = curate(rows, quotas, 1)
        self.assertEqual(len({row.split for row in selected}), 1)
        self.assertIn("near_duplicate_cross_split", {row["reason"] for row in rejected})
        self.assertEqual(sum(shortfall.values()), 1)

    def test_no_target_has_independent_quota(self):
        rows = [candidate("target", "development", "a", "0" * 16), candidate("empty", "development", "b", "f" * 16, target=False)]
        quotas = {("simple", "target", "development"): 1, ("all", "no_target", "development"): 1}
        selected, _, _, coverage, shortfall = curate(rows, quotas, 0)
        self.assertEqual(len(selected), 2)
        self.assertEqual(coverage["all/no_target/development"], 1)
        self.assertFalse(any(shortfall.values()))

    def test_near_duplicate_does_not_chain_through_intermediate_hash(self):
        rows = [
            candidate("a", "development", "a", "0000000000000000"),
            candidate("b", "development", "b", "0000000000000003"),
            candidate("c", "development", "c", "000000000000000f"),
        ]
        selected, _, clusters, _, _ = curate(rows, {("simple", "target", "development"): 3}, 2)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(clusters), 2)


if __name__ == "__main__":
    unittest.main()
