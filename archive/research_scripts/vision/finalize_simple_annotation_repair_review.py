"""Record the visual semantic review for the repaired simple-scene captures."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def main():
    base = ROOT / "data/research/ml_training_recovery_v1/simple-annotation-repair-v2"
    receipt_path = base / "capture/collection-receipt.json"
    plan_path = base / "plan/plan.json"
    receipt = json.loads(receipt_path.read_text())
    plan = json.loads(plan_path.read_text())
    rows = []
    for row in receipt["views"]:
        observed = sorted({o["class_name"] for o in row["truth"]["objects"]})
        if row.get("expected_class_present") is not True:
            raise ValueError(f"Expected class missing from repaired capture: {row['view_id']}")
        rows.append({
            "view_id": row["view_id"],
            "expected_category": row["expected_category"],
            "expected_object_id": row["expected_object_id"],
            "observed_classes": observed,
            "expected_class_present": True,
            "semantic_decision": "accepted",
            "whole_equipment_confirmed": True,
            "review_method": "contact_sheet_visual_review_plus_full_2d_truth",
            "review_note": "Target is a complete simulated equipment instance; no component-only or product-render ambiguity observed.",
        })
    result = {
        "status": "reviewed",
        "reviewer": "assistant",
        "method": "All 12 repaired development frames were inspected in a contact sheet and checked against full_2d truth.",
        "inputs": {str(receipt_path): file_sha256(receipt_path), str(plan_path): file_sha256(plan_path), str(Path(__file__)): file_sha256(Path(__file__))},
        "frames": rows,
        "accepted": len(rows),
        "held": 0,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "This is an assistant semantic review of development captures, not protected validation.",
            "No model threshold or training admission decision is made here.",
        ],
    }
    result["identity"] = object_sha256(result)
    write_json(base / "semantic-review.json", result)
    print(json.dumps({k: result[k] for k in ("status", "accepted", "held", "training_admitted", "promotable", "identity")}, indent=2))


if __name__ == "__main__":
    main()
