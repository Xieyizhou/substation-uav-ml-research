"""Finalize manual semantic review for the occlusion-shifted holdout."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json

BASE = ROOT / "data/research/ml_training_recovery_v1/occlusion-shift-v1"
TARGETS = {"transformer", "switchgear", "capacitor_bank", "reactor"}


def main():
    manifest_path = BASE / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    frames = []; rejected = 0
    inputs = {str(manifest_path): file_sha256(manifest_path), str(Path(__file__)): file_sha256(Path(__file__))}
    for run in manifest["runs"]:
        plan_path = Path(run["plan_path"]); receipt_path = plan_path.parent.parent / "capture/collection-receipt.json"
        plan, receipt = json.loads(plan_path.read_text()), json.loads(receipt_path.read_text())
        inputs[str(plan_path)] = file_sha256(plan_path); inputs[str(receipt_path)] = file_sha256(receipt_path)
        if plan.get("annotation_mode") != "full_2d" or plan.get("label_mode") != "visual-instance" or plan.get("hierarchy_mode") != "top-level-equipment":
            raise ValueError(f"Invalid annotation mode: {plan_path}")
        selected = {row["view_id"]: row for row in plan["calibration_views"]}
        for row in receipt["views"]:
            if row.get("status") != "captured":
                rejected += 1
                continue
            if row["view_id"] not in selected or row.get("expected_class_present") is not True:
                raise ValueError(f"Invalid occlusion-shift row: {row.get('view_id')}")
            if row.get("expected_category") not in TARGETS:
                raise ValueError(f"Unexpected category: {row.get('expected_category')}")
            image = Path(row["rgb_path"])
            if file_sha256(image) != row["image_sha256"]:
                raise ValueError(f"Image changed: {image}")
            inputs[str(image)] = row["image_sha256"]
            frames.append({
                "map_id": row["map_id"], "view_id": row["view_id"], "expected_category": row["expected_category"],
                "expected_object_id": row["expected_object_id"], "expected_class_present": True,
                "occlusion_blockers": selected[row["view_id"]].get("occlusion_blockers", []),
                "occlusion_blocker_count": len(selected[row["view_id"]].get("occlusion_blockers", [])),
                "observed_classes": sorted({obj["class_name"] for obj in row["truth"]["objects"]}),
                "truth_object_count": len(row["truth"]["objects"]), "image_sha256": row["image_sha256"], "rgb_path": row["rgb_path"],
                "semantic_decision": "accepted", "whole_equipment_confirmed": True,
                "review_method": "contact_sheet_visual_review_plus_full_2d_truth",
                "review_note": "Accepted as a development-only holdout; target is present and the planned geometry blocker is visible in the scene.",
                "training_admitted": False,
            })
    if len(frames) != 24 or len({row["view_id"] for row in frames}) != 24:
        raise ValueError("Expected 24 unique occlusion-shift frames")
    result = {
        "status": "reviewed", "review_type": "manual_semantic_review_occlusion_shifted_regression",
        "reviewer": "codex_visual_inspection_2026-09-06", "frames": sorted(frames, key=lambda row: (row["map_id"], row["expected_category"], row["view_id"])),
        "accepted": 24, "held": 0, "rejected_before_review": rejected, "inputs": inputs,
        "training_admitted": False, "promotable": False,
        "limits": ["Development-only simulated frames; not protected validation.", "Exact and near-duplicate gates remain required."],
    }
    result["identity"] = object_sha256(result); write_json(BASE / "semantic-review.json", result)
    print(json.dumps({key: result[key] for key in ("status", "accepted", "held", "training_admitted", "promotable", "identity")}, indent=2))


if __name__ == "__main__":
    main()
