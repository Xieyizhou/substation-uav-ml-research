"""Evaluate current diagnostic weights on targeted scale/map coverage frames."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.evaluate_stratified_recheck import score, TARGETS


BASE = ROOT / "data/research/ml_training_recovery_v1"
TARGETED = BASE / "targeted-coverage-v1"
TRAINING = BASE / "expanded-stratified-v1"


def main():
    from ultralytics import YOLO
    rows = []
    for map_id in ("medium", "complex"):
        receipt = json.loads((TARGETED / map_id / "capture/collection-receipt.json").read_text())
        rows.extend({"map_id": map_id, "view_id": row["view_id"], "path": row["rgb_path"], "truth": row["truth"]["objects"], "expected": row["expected_category"]} for row in receipt["views"])
    if len(rows) != 24: raise ValueError("Expected 24 targeted frames")
    weights = {"positive_only": BASE / "memorization-v1/fit-batchmatched/weights/last.pt", "negative_augmented": BASE / "negative-ab-v1/fit/weights/last.pt"}
    for seed in (7, 17, 27): weights[f"expanded_uniform_{seed}"] = TRAINING / f"uniform-{seed}/weights/last.pt"
    results = {}; inputs = {str(Path(__file__)): file_sha256(Path(__file__)), str(TARGETED / "semantic-review.json"): file_sha256(TARGETED / "semantic-review.json"), str(TARGETED / "dedup-audit.json"): file_sha256(TARGETED / "dedup-audit.json")}
    for name, weight in weights.items():
        model = YOLO(str(weight)); scored = []
        for row in rows:
            path = Path(row["path"]); inputs[str(path)] = file_sha256(path); scored.append({**row, "score": score(model, path, row["truth"], row["expected"])})
        results[name] = {"weights_sha256": file_sha256(weight), "rows": scored}
    summary = {}
    for name, result in results.items():
        summary[name] = {}
        for group, group_rows in [("all", result["rows"]), ("medium", [r for r in result["rows"] if r["map_id"] == "medium"]), ("complex", [r for r in result["rows"] if r["map_id"] == "complex"])]:
            expected = [r for r in group_rows if r["expected"] in TARGETS]
            summary[name][group] = {"frames": len(group_rows), "truth_objects": sum(r["score"]["truth_count"] for r in group_rows), "matched_hits": sum(r["score"]["matched_hits"] for r in group_rows), "expected_class_hits": sum(bool(r["score"]["expected_class_hit"]) for r in expected), "expected_class_frames": len(expected), "predictions": sum(r["score"]["predictions"] for r in group_rows)}
    report = {"status": "targeted_coverage_evaluation_complete", "rows": len(rows), "results": results, "summary": summary, "inputs": inputs, "training_admitted": False, "promotable": False, "limits": ["Development-only targeted evaluation.", "No threshold selection or model promotion."]}
    report["identity"] = object_sha256(report); write_json(TARGETED / "evaluation.json", report); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
