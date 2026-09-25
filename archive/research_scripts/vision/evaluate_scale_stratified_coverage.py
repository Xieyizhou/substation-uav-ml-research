"""Evaluate existing development weights on scale-stratified coverage frames."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.evaluate_stratified_recheck import TARGETS, score


BASE = ROOT / "data/research/ml_training_recovery_v1"
DATA = BASE / "scale-stratified-v1"
DISTANCE_BINS = {"near": (5.0, 9.99), "mid": (10.0, 14.99), "far": (15.0, 30.0)}


def distance_bin(distance):
    for name, (low, high) in DISTANCE_BINS.items():
        if low <= distance <= high:
            return name
    raise ValueError(f"Distance outside frozen bins: {distance}")


def load_rows():
    rows = []
    for map_id in ("simple", "medium", "complex"):
        plan_path = DATA / map_id / "plan/plan.json"
        receipt_path = DATA / map_id / "capture/collection-receipt.json"
        plan = json.loads(plan_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        selected = {row["view_id"]: row for row in plan["calibration_views"]}
        for row in receipt["views"]:
            spec = selected[row["view_id"]]
            rows.append(
                {
                    "map_id": map_id,
                    "view_id": row["view_id"],
                    "path": row["rgb_path"],
                    "truth": row["truth"]["objects"],
                    "expected": row["expected_category"],
                    "distance": spec["distance"],
                    "distance_bin": distance_bin(spec["distance"]),
                }
            )
    if len(rows) != 36:
        raise ValueError(f"Expected 36 scale-stratified rows, got {len(rows)}")
    return rows


def summarize(rows):
    expected = [row for row in rows if row["expected"] in TARGETS]
    return {
        "frames": len(rows),
        "truth_objects": sum(row["score"]["truth_count"] for row in rows),
        "matched_hits": sum(row["score"]["matched_hits"] for row in rows),
        "predictions": sum(row["score"]["predictions"] for row in rows),
        "expected_class_hits": sum(bool(row["score"]["expected_class_hit"]) for row in expected),
        "expected_class_frames": len(expected),
    }


def main():
    from ultralytics import YOLO

    rows = load_rows()
    weights = {
        "positive_only": BASE / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": BASE / "negative-ab-v1/fit/weights/last.pt",
    }
    expanded = BASE / "expanded-stratified-v1"
    for seed in (7, 17, 27):
        weights[f"expanded_uniform_{seed}"] = expanded / f"uniform-{seed}/weights/last.pt"

    inputs = {
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(DATA / "manifest.json"): file_sha256(DATA / "manifest.json"),
        str(DATA / "semantic-review.json"): file_sha256(DATA / "semantic-review.json"),
        str(DATA / "dedup-audit.json"): file_sha256(DATA / "dedup-audit.json"),
    }
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
        groups = {"all": result["rows"]}
        groups.update({f"map:{map_id}": [row for row in result["rows"] if row["map_id"] == map_id] for map_id in ("simple", "medium", "complex")})
        groups.update({f"distance:{bin_name}": [row for row in result["rows"] if row["distance_bin"] == bin_name] for bin_name in DISTANCE_BINS})
        groups.update({f"category:{category}": [row for row in result["rows"] if row["expected"] == category] for category in sorted(TARGETS)})
        summary[name] = {group: summarize(group_rows) for group, group_rows in groups.items()}

    report = {
        "status": "scale_stratified_coverage_evaluation_complete",
        "rows": len(rows),
        "distance_bins": DISTANCE_BINS,
        "results": results,
        "summary": summary,
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only baseline evaluation; no protected labels or metrics.",
            "No threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(DATA / "evaluation.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
