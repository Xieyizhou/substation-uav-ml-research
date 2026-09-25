"""Freeze the reviewed 48-frame positive bridge supplement as a non-admitted ledger."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE


def load(path):
    return json.loads(path.read_text())


def index_capture(run_root, capture_dir):
    indexed = {}
    inputs = {}
    for receipt_path in run_root.glob(f"runs/*/{capture_dir}/collection-receipt.json"):
        receipt = load(receipt_path)
        plan_path = receipt_path.parent.parent / "plan/plan.json"
        plan = load(plan_path)
        if receipt["status"] != "complete_pending_review":
            raise ValueError(f"Capture is not complete: {receipt_path}")
        if receipt["actual_annotation_mode"] != "full_2d":
            raise ValueError(f"Actual annotation mode regressed: {receipt_path}")
        if receipt["collection_checks"]["check_version"] != "canonical-collection-gates-v2":
            raise ValueError(f"Collection gate version regressed: {receipt_path}")
        plans = {row["view_id"]: row for row in plan["calibration_views"]}
        for view in receipt["views"]:
            if view["status"] != "captured":
                raise ValueError(f"Frame is not captured: {view['view_id']}")
            checks = view["target_checks"]
            if not checks["category_present"] or not checks["planned_instance_present"]:
                raise ValueError(f"Target gate failed: {view['view_id']}")
            indexed[view["view_id"]] = (receipt_path, receipt, plan_path, plan, plans[view["view_id"]], view)
        inputs[str(receipt_path)] = file_sha256(receipt_path)
        inputs[str(plan_path)] = file_sha256(plan_path)
    return indexed, inputs


def main():
    matrix_path = BASE / "matrix.json"
    matrix = load(matrix_path)
    pilot_completion_path = BASE / "pilot-v1/completion.json"
    pilot_review_path = BASE / "pilot-v1/review-v1/semantic-review.json"
    pilot_dedup_path = BASE / "pilot-v1/review-v1/dedup-audit.json"
    remaining_review_path = BASE / "remaining-positive-v1/review-v1/semantic-review.json"
    remaining_dedup_path = BASE / "remaining-positive-v1/review-v1/dedup-audit.json"
    pilot_completion = load(pilot_completion_path)
    reviews = [load(pilot_review_path), load(remaining_review_path)]
    dedups = [load(pilot_dedup_path), load(remaining_dedup_path)]
    if pilot_completion["status"] != "pilot_passed_ready_for_remaining_positive_capture":
        raise ValueError("Pilot is not finalized")
    if sum(review["accepted"] for review in reviews) != 48 or any(review["held"] for review in reviews):
        raise ValueError("The positive review set is not 48/48 accepted")
    if any(audit["status"] != "passed" for audit in dedups):
        raise ValueError("A dedup audit is not passed")

    capture_index = {}
    inputs = {str(matrix_path): file_sha256(matrix_path)}
    for root, capture_dir in ((BASE / "pilot-v1", "capture-retry-1"), (BASE / "remaining-positive-v1", "capture")):
        indexed, capture_inputs = index_capture(root, capture_dir)
        if set(capture_index) & set(indexed):
            raise ValueError("A captured view_id occurs in both capture phases")
        capture_index.update(indexed)
        inputs.update(capture_inputs)

    dedup_by_view = {row["view_id"]: row for audit in dedups for row in audit["decisions"]}
    review_rows = [row for review in reviews for row in review["frames"]]
    if set(capture_index) != {row["view_id"] for row in review_rows} or len(review_rows) != 48:
        raise ValueError("Capture and explicit review membership differ")

    ledger = []
    for review in review_rows:
        receipt_path, receipt, plan_path, plan, planned, captured = capture_index[review["view_id"]]
        if file_sha256(Path(review["image_path"])) != review["image_sha256"]:
            raise ValueError(f"Reviewed image changed: {review['view_id']}")
        dedup = dedup_by_view[review["view_id"]]
        if not dedup["status"].startswith("accepted_no_"):
            raise ValueError(f"Unresolved duplicate: {review['view_id']}")
        ledger.append({
            "view_id": review["view_id"],
            "pair_id": planned["pair_id"],
            "derivation_group": planned["derivation_group"],
            "data_role": "new_training_candidate",
            "map_id": captured["map_id"],
            "asset_id": captured["expected_object_id"],
            "device_instance": captured["expected_object_id"],
            "category": captured["expected_category"],
            "variant": plan["variant"],
            "planned_pose": {"position": planned["position"], "orientation": planned["orientation"]},
            "actual_pose": {"position": captured["actual_pose"]["position"], "orientation": captured["actual_pose"]["orientation"]},
            "image_path": review["image_path"],
            "image_sha256": review["image_sha256"],
            "label_sha256": review["truth_sha256"],
            "world_sha256": receipt["world_sha256"],
            "sensor_sha256": plan["files"]["sensor_source.sdf"],
            "plan_identity": plan["identity"],
            "receipt_identity": receipt["identity"],
            "receipt_path": str(receipt_path),
            "annotation": {
                "actual_mode": receipt["actual_annotation_mode"],
                "label_mode": receipt["collection_checks"]["label_mode"],
                "hierarchy_mode": receipt["collection_checks"]["hierarchy_mode"],
                "check_version": receipt["collection_checks"]["check_version"],
            },
            "instance_checks": captured["target_checks"],
            "review": {
                "decision": review["decision"],
                "reason": review["reason"],
                "nature": "AI-assisted",
                "reviewed_at": review["reviewed_at"],
                "visibility_status": review["visibility_status"],
                "truncation_status": review["truncation_status"],
            },
            "duplicate_status": dedup["status"],
            "training_admitted": False,
            "promotable": False,
        })

    pair_counts = Counter(row["pair_id"] for row in ledger)
    if len(pair_counts) != 16 or set(pair_counts.values()) != {3}:
        raise ValueError("Expected 16 complete three-variant lineage groups")
    result = {
        "schema_version": 1,
        "status": "positive_supplement_frozen_pending_negative_completion",
        "matrix_identity": matrix["identity"],
        "frame_count": len(ledger),
        "independent_pose_groups": len(pair_counts),
        "variant_counts": dict(Counter(row["variant"] for row in ledger)),
        "category_counts": dict(Counter(row["category"] for row in ledger)),
        "frames": sorted(ledger, key=lambda row: (row["pair_id"], row["variant"])),
        "negative_redesign_status": "pending_plan_materialization_and_source_validation",
        "unseen_scene_status": "sealed_not_evaluated",
        "inputs": {
            **inputs,
            str(pilot_completion_path): file_sha256(pilot_completion_path),
            str(pilot_review_path): file_sha256(pilot_review_path),
            str(pilot_dedup_path): file_sha256(pilot_dedup_path),
            str(remaining_review_path): file_sha256(remaining_review_path),
            str(remaining_dedup_path): file_sha256(remaining_dedup_path),
            str(Path(__file__)): file_sha256(Path(__file__)),
        },
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "frozen-positive-ledger.json", result)
    print(json.dumps({k: result[k] for k in ("status", "frame_count", "independent_pose_groups", "variant_counts", "category_counts", "identity")}, indent=2))


if __name__ == "__main__":
    main()
