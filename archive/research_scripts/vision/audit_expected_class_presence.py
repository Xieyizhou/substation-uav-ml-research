"""Audit whether diagnostic full_2d captures contain their planned target class."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record

TARGET_CLASSES = {"transformer", "switchgear", "capacitor_bank", "reactor"}
RECEIPT_ROOTS = (
    ROOT / "data/research/ml_training_recovery_v1/paired-calibration-v1",
    ROOT / "data/research/ml_training_recovery_v1/control-matrix-capture-v1",
    ROOT / "data/research/ml_training_recovery_v1/cross-scene-recheck-v1",
    ROOT / "data/research/ml_training_recovery_v1/simple-independent-scene-v1",
    ROOT / "data/research/ml_training_recovery_v1/simple-annotation-repair-v2",
    ROOT / "data/research/ml_training_recovery_v1/stratified-expansion-v1",
    ROOT / "data/research/ml_training_recovery_v1/targeted-coverage-v1",
)


def classify(row):
    expected = row.get("expected_category")
    observed = sorted({o["class_name"] for o in row.get("truth", {}).get("objects", [])})
    target_expected = expected in TARGET_CLASSES
    present = expected in observed if target_expected else None
    if target_expected and not present:
        status = "expected_target_absent"
    elif target_expected:
        status = "observable_expected_target"
    else:
        status = "non_target_or_background_plan"
    return {
        "view_id": row.get("view_id"),
        "map_id": row.get("map_id"),
        "expected_category": expected,
        "expected_class_present": present,
        "observed_classes": observed,
        "status": status,
        "rgb_path": row.get("rgb_path"),
        "image_sha256": row.get("image_sha256"),
    }


def main():
    inputs = {str(Path(__file__)): file_sha256(Path(__file__))}
    rows = []
    receipts = []
    for root in RECEIPT_ROOTS:
        if not root.exists():
            continue
        for receipt_path in sorted(root.rglob("collection-receipt.json")):
            receipt = read_record(receipt_path)
            inputs[str(receipt_path)] = file_sha256(receipt_path)
            plan_path = receipt_path.parent.parent / "plan" / "plan.json"
            plan = {}
            if plan_path.exists():
                inputs[str(plan_path)] = file_sha256(plan_path)
                plan = read_record(plan_path)
                assert receipt.get("plan_identity") == plan.get("identity")
            captured = [classify(row) for row in receipt.get("views", []) if row.get("status") == "captured"]
            legacy_annotation_mode = not (
                plan.get("label_mode") == "visual-instance"
                and plan.get("hierarchy_mode") == "top-level-equipment"
            )
            for row in captured:
                row["label_mode"] = plan.get("label_mode")
                row["hierarchy_mode"] = plan.get("hierarchy_mode")
                row["legacy_annotation_mode"] = legacy_annotation_mode
            rows.extend(captured)
            receipts.append({"path": str(receipt_path), "captured": len(captured), "status": receipt.get("status"), "label_mode": plan.get("label_mode"), "hierarchy_mode": plan.get("hierarchy_mode"), "legacy_annotation_mode": legacy_annotation_mode})

    by_category = defaultdict(Counter)
    by_map = defaultdict(Counter)
    by_annotation_mode = defaultdict(Counter)
    for row in rows:
        key = row["expected_category"] or "missing"
        by_category[key][row["status"]] += 1
        by_map[row["map_id"] or "missing"][row["status"]] += 1
        by_annotation_mode["legacy" if row["legacy_annotation_mode"] else "visual_instance_top_level"][row["status"]] += 1
    report = {
        "status": "expected_class_presence_audit_complete",
        "receipts": receipts,
        "captured": len(rows),
        "rows": rows,
        "by_expected_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "by_map": {k: dict(v) for k, v in sorted(by_map.items())},
        "by_annotation_mode": {k: dict(v) for k, v in sorted(by_annotation_mode.items())},
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "This audit checks class presence in diagnostic full_2d truth only; it does not certify visibility quality or annotation completeness.",
            "Expected-target-absent rows are holds and must not enter a recall denominator or be treated as hard negatives without review.",
            "All source captures remain development diagnostics.",
        ],
    }
    report["identity"] = object_sha256(report)
    out = ROOT / "data/research/ml_training_recovery_v1/expected-class-presence-audit-v1/report.json"
    write_json(out, report)
    print(json.dumps({k: report[k] for k in ("status", "captured", "by_expected_category", "by_map", "by_annotation_mode")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
