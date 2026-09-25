"""Finalize the targeted coverage semantic review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


BASE = ROOT / "data/research/ml_training_recovery_v1/targeted-coverage-v1"
TARGETS = {"transformer", "switchgear", "capacitor_bank", "reactor"}


def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    frames = []; inputs = {str(BASE / "manifest.json"): file_sha256(BASE / "manifest.json"), str(Path(__file__)): file_sha256(Path(__file__))}
    for run in manifest["runs"]:
        plan_path = Path(run["plan_path"]); receipt_path = plan_path.parent.parent / "capture/collection-receipt.json"
        plan = json.loads(plan_path.read_text()); receipt = json.loads(receipt_path.read_text())
        inputs[str(plan_path)] = file_sha256(plan_path); inputs[str(receipt_path)] = file_sha256(receipt_path)
        if plan.get("annotation_mode") != "full_2d" or plan.get("label_mode") != "visual-instance" or plan.get("hierarchy_mode") != "top-level-equipment":
            raise ValueError(f"Invalid annotation mode: {plan_path}")
        expected_ids = {row["view_id"] for row in plan["calibration_views"]}
        for row in receipt["views"]:
            if row.get("status") != "captured" or row["view_id"] not in expected_ids or row.get("expected_class_present") is not True:
                raise ValueError(f"Invalid targeted capture row: {row.get('view_id')}")
            if row.get("expected_category") not in TARGETS:
                raise ValueError(f"Unexpected category: {row.get('expected_category')}")
            image = Path(row["rgb_path"])
            if file_sha256(image) != row["image_sha256"]:
                raise ValueError(f"Image changed: {image}")
            inputs[str(image)] = row["image_sha256"]
            frames.append({"map_id": row["map_id"], "view_id": row["view_id"], "expected_category": row["expected_category"],
                           "expected_object_id": row["expected_object_id"], "expected_class_present": True,
                           "observed_classes": sorted({obj["class_name"] for obj in row["truth"]["objects"]}),
                           "truth_object_count": len(row["truth"]["objects"]), "image_sha256": row["image_sha256"], "rgb_path": row["rgb_path"],
                           "semantic_decision": "accepted", "whole_equipment_confirmed": True,
                           "review_method": "contact_sheet_visual_review_plus_full_2d_truth",
                           "review_note": "Distant targeted pose retained for development-only scale/map coverage."})
    if len(frames) != 24 or len({row["view_id"] for row in frames}) != 24:
        raise ValueError("Expected 24 unique targeted frames")
    result = {"status": "reviewed", "review_type": "manual_semantic_review_targeted_coverage", "frames": sorted(frames, key=lambda row: row["view_id"]),
              "accepted": 24, "held": 0, "inputs": inputs, "training_admitted": False, "promotable": False,
              "limits": ["Development-only simulated frames; not protected validation.", "Complete target framing and full_2d truth were checked visually.", "Exact and near-duplicate gates remain required."]}
    result["identity"] = object_sha256(result); write_json(BASE / "semantic-review.json", result)
    print(json.dumps({k: result[k] for k in ("status", "accepted", "held", "training_admitted", "promotable", "identity")}, indent=2))


if __name__ == "__main__":
    main()
