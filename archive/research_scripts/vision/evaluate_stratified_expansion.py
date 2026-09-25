"""Evaluate existing diagnostic weights on the hash-clean expansion frames."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.evaluate_stratified_recheck import score


TARGETS = {"transformer", "switchgear", "capacitor_bank", "reactor"}
BASE = ROOT / "data/research/ml_training_recovery_v1"
EXPANSION = BASE / "stratified-expansion-v1"


def load_rows():
    rows = []
    for map_id in ("simple", "medium", "complex"):
        receipt_path = EXPANSION / map_id / "capture/collection-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        for row in receipt["views"]:
            rows.append({
                "map_id": map_id,
                "view_id": row["view_id"],
                "path": row["rgb_path"],
                "truth": row["truth"]["objects"],
                "expected": row["expected_category"],
            })
    if len(rows) != 36:
        raise ValueError(f"Expected 36 expansion rows, got {len(rows)}")
    return rows


def main():
    from ultralytics import YOLO

    rows = load_rows()
    weights = {
        "positive_only": BASE / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": BASE / "negative-ab-v1/fit/weights/last.pt",
    }
    protocol = json.loads((BASE / "stratified-sampling-v1/protocol.json").read_text())
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:
            weights[f"{arm}-{seed}"] = BASE / "stratified-sampling-v1" / f"{arm}-{seed}" / "weights/last.pt"

    inputs = {str(Path(__file__)): file_sha256(Path(__file__)), str(BASE / "stratified-sampling-v1/protocol.json"): file_sha256(BASE / "stratified-sampling-v1/protocol.json")}
    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        scored = []
        for row in rows:
            path = Path(row["path"])
            inputs[str(path)] = file_sha256(path)
            scored.append({**row, "score": score(model, path, row["truth"], row["expected"])})
        results[name] = {"weights_sha256": file_sha256(weight), "rows": scored}

    summary = {}
    for name, result in results.items():
        by_group = {}
        for key, group_rows in [("all", result["rows"])] + [(f"map:{m}", [r for r in result["rows"] if r["map_id"] == m]) for m in ("simple", "medium", "complex")]:
            expected = [r for r in group_rows if r["expected"] in TARGETS]
            by_group[key] = {
                "frames": len(group_rows),
                "truth_objects": sum(r["score"]["truth_count"] for r in group_rows),
                "matched_hits": sum(r["score"]["matched_hits"] for r in group_rows),
                "expected_class_hits": sum(bool(r["score"]["expected_class_hit"]) for r in expected),
                "expected_class_frames": len(expected),
                "predictions": sum(r["score"]["predictions"] for r in group_rows),
            }
        by_category = {}
        for category in sorted(TARGETS):
            group_rows = [r for r in result["rows"] if r["expected"] == category]
            by_category[category] = {
                "frames": len(group_rows),
                "matched_hits": sum(r["score"]["matched_hits"] for r in group_rows),
                "expected_class_hits": sum(bool(r["score"]["expected_class_hit"]) for r in group_rows),
            }
        summary[name] = {"by_group": by_group, "by_category": by_category}

    report = {
        "status": "stratified_expansion_evaluation_complete",
        "rows": len(rows),
        "results": results,
        "summary": summary,
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only expansion evaluation; no protected labels or metrics.",
            "Frames are hash-clean against the historical non-protected receipt scope and were semantically reviewed.",
            "No threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(EXPANSION / "evaluation.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
