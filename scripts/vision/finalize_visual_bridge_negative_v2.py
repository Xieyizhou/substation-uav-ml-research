"""Freeze the reviewed 24-frame canonical-only negative supplement."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE


def main():
    matrix_path = BASE / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    review_paths = [BASE / "pilot-v1/review-v1/semantic-review.json", BASE / "remaining-v1/review-v1/semantic-review.json"]
    dedup_paths = [BASE / "pilot-v1/review-v1/dedup-audit.json", BASE / "remaining-v1/review-v1/dedup-audit.json"]
    reviews = [json.loads(path.read_text()) for path in review_paths]
    dedups = [json.loads(path.read_text()) for path in dedup_paths]
    if sum(row["accepted"] for row in reviews) != 24 or any(row["held"] for row in reviews):
        raise ValueError("Explicit negative review is not 24/24 accepted")
    if any(row["status"] != "passed" for row in dedups):
        raise ValueError("Negative dedup audit is incomplete")
    dedup_by_view = {row["view_id"]: row for audit in dedups for row in audit["decisions"]}
    capture = {}
    inputs = {str(matrix_path): file_sha256(matrix_path), str(Path(__file__)): file_sha256(Path(__file__))}
    for phase in ("pilot-v1", "remaining-v1"):
        for receipt_path in (BASE / phase / "runs").glob("*/capture/collection-receipt.json"):
            receipt = read_record(receipt_path)
            plan_path = receipt_path.parent.parent / "plan/plan.json"
            plan = read_record(plan_path)
            if receipt["status"] != "complete_pending_review" or receipt["actual_annotation_mode"] != "full_2d":
                raise ValueError(f"Capture gate incomplete: {receipt_path}")
            planned = {row["view_id"]: row for row in plan["calibration_views"]}
            for view in receipt["views"]:
                if view["status"] != "captured" or view["truth"]["objects"] or view["target_checks"]["observed_instance_labels"]:
                    raise ValueError(f"Non-empty negative frame: {view['view_id']}")
                capture[view["view_id"]] = (receipt_path, receipt, plan_path, plan, planned[view["view_id"]], view)
            inputs[str(receipt_path)] = file_sha256(receipt_path)
            inputs[str(plan_path)] = file_sha256(plan_path)
    reviewed = [row for review in reviews for row in review["frames"]]
    if len(capture) != 24 or set(capture) != {row["view_id"] for row in reviewed}:
        raise ValueError("Review and capture membership differ")
    frames = []
    for review in reviewed:
        receipt_path, receipt, plan_path, plan, planned, view = capture[review["view_id"]]
        duplicate = dedup_by_view[review["view_id"]]
        if duplicate["status"] != "accepted_no_independent_duplicate" or file_sha256(Path(review["image_path"])) != review["image_sha256"]:
            raise ValueError(f"Review or duplicate evidence changed: {review['view_id']}")
        frames.append({
            "view_id": review["view_id"], "pair_id": planned["pair_id"], "derivation_group": planned["derivation_group"],
            "data_role": "new_training_candidate", "map_id": view["map_id"], "source_layout_id": plan["source_layout_id"],
            "map_layout_id": plan["derived_layout_id"], "subject_family": planned["subject_family"],
            "subject_instance": planned["object_id"], "lighting_id": planned["lighting_id"],
            "planned_pose": {"position": planned["position"], "orientation": planned["orientation"]},
            "actual_pose": {"position": view["actual_pose"]["position"], "orientation": view["actual_pose"]["orientation"]},
            "image_path": review["image_path"], "image_sha256": review["image_sha256"], "label_sha256": review["truth_sha256"],
            "truth_object_count": 0, "world_sha256": receipt["world_sha256"], "sensor_sha256": plan["files"]["sensor_source.sdf"],
            "plan_identity": plan["identity"], "receipt_identity": receipt["identity"], "receipt_path": str(receipt_path),
            "annotation": {"actual_mode": receipt["actual_annotation_mode"], "label_mode": receipt["collection_checks"]["label_mode"],
                           "hierarchy_mode": receipt["collection_checks"]["hierarchy_mode"], "check_version": receipt["check_version"]},
            "review": {"decision": review["decision"], "reason": review["reason"], "nature": "AI-assisted",
                       "reviewed_at": review["reviewed_at"], "target_exclusion_status": review["target_exclusion_status"]},
            "duplicate_status": duplicate["status"], "training_admitted": False, "promotable": False,
        })
    pair_counts = Counter(row["pair_id"] for row in frames)
    if len(pair_counts) != 12 or set(pair_counts.values()) != {2}:
        raise ValueError("Negative lighting pairs are incomplete")
    result = {"schema_version": 1, "status": "frozen_nonadmitted", "matrix_identity": matrix["identity"],
              "frame_count": 24, "independent_pose_groups": 12,
              "lighting_counts": dict(Counter(row["lighting_id"] for row in frames)),
              "family_counts": dict(sorted(Counter(row["subject_family"] for row in frames).items())),
              "frames": sorted(frames, key=lambda row: (row["pair_id"], row["lighting_id"])),
              "off_scope_background_package_used": False, "inputs": {**inputs, **{str(p): file_sha256(p) for p in review_paths + dedup_paths}},
              "unseen_scene_status": "sealed_not_evaluated", "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result)
    write_json(BASE / "frozen-negative-ledger.json", result)
    positive_path = BASE.parent / "frozen-positive-ledger.json"
    positive = json.loads(positive_path.read_text())
    completion = {"schema_version": 1, "status": "visual_bridge_supplement_frozen_pending_training_admission_decision",
                  "positive_ledger_identity": positive["identity"], "positive_frames": positive["frame_count"],
                  "negative_ledger_identity": result["identity"], "negative_frames": result["frame_count"],
                  "total_frames": positive["frame_count"] + result["frame_count"], "independent_positive_pose_groups": 16,
                  "independent_negative_pose_groups": 12, "off_scope_background_package_used": False,
                  "unseen_scene_status": "sealed_not_evaluated", "training_started": False,
                  "training_admitted": False, "promotable": False,
                  "inputs": {str(positive_path): file_sha256(positive_path), str(BASE / "frozen-negative-ledger.json"): file_sha256(BASE / "frozen-negative-ledger.json")}}
    completion["identity"] = object_sha256(completion)
    write_json(BASE.parent / "supplement-completion.json", completion)
    print(json.dumps({"negative_status": result["status"], "negative_identity": result["identity"],
                      "completion_status": completion["status"], "total_frames": completion["total_frames"],
                      "completion_identity": completion["identity"]}, indent=2))


if __name__ == "__main__":
    main()
