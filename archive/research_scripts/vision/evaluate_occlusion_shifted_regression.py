"""Evaluate development weights on the occlusion-shifted regression holdout."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.evaluate_scale_expansion_recheck import score_at_640, summarize

BASE = ROOT / "data/research/ml_training_recovery_v1"
DATA = BASE / "occlusion-shift-v1"


def load_rows():
    rows = []
    for map_id in ("medium", "complex"):
        plan = json.loads((DATA / map_id / "plan/plan.json").read_text())
        receipt = json.loads((DATA / map_id / "capture/collection-receipt.json").read_text())
        selected = {row["view_id"]: row for row in plan["calibration_views"]}
        for row in receipt["views"]:
            if row.get("status") != "captured":
                continue
            spec = selected[row["view_id"]]
            rows.append({
                "map_id": map_id, "view_id": row["view_id"], "path": row["rgb_path"],
                "truth": row["truth"]["objects"], "expected": row["expected_category"],
                "distance": spec["distance"], "distance_bin": "near" if spec["distance"] < 10 else "mid" if spec["distance"] < 15 else "far",
                "occlusion_blockers": spec.get("occlusion_blockers", []),
                "occlusion_blocker_count": len(spec.get("occlusion_blockers", [])),
            })
    if len(rows) != 24:
        raise ValueError(f"Expected 24 occlusion-shift rows, got {len(rows)}")
    return rows


def main():
    from ultralytics import YOLO
    rows = load_rows()
    weights = {
        "positive_only": BASE / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": BASE / "negative-ab-v1/fit/weights/last.pt",
    }
    for seed in (7, 17, 27):
        weights[f"expanded_uniform_{seed}"] = BASE / f"expanded-stratified-v1/uniform-{seed}/weights/last.pt"
        weights[f"scale100_{seed}"] = BASE / f"scale-expansion-v1/uniform-{seed}/weights/last.pt"
        weights[f"scale50_{seed}"] = BASE / f"scale-budget-ab-v1/steps50-{seed}/weights/last.pt"
    inputs = {
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(DATA / "manifest.json"): file_sha256(DATA / "manifest.json"),
        str(DATA / "semantic-review.json"): file_sha256(DATA / "semantic-review.json"),
        str(DATA / "dedup-audit.json"): file_sha256(DATA / "dedup-audit.json"),
    }
    for row in rows: inputs[str(Path(row["path"]))] = file_sha256(Path(row["path"]))
    for weight in weights.values(): inputs[str(weight)] = file_sha256(weight)
    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        scored = [{**row, "score": score_at_640(model, Path(row["path"]), row["truth"], row["expected"])} for row in rows]
        results[name] = {"weights_sha256": file_sha256(weight), "rows": scored}
    summary = {}
    for name, result in results.items():
        groups = {"all": result["rows"]}
        groups.update({f"map:{map_id}": [row for row in result["rows"] if row["map_id"] == map_id] for map_id in ("medium", "complex")})
        groups.update({f"distance:{bin_name}": [row for row in result["rows"] if row["distance_bin"] == bin_name] for bin_name in ("near", "mid", "far")})
        groups.update({f"blockers:{count}": [row for row in result["rows"] if row["occlusion_blocker_count"] == count] for count in (1, 2)})
        groups.update({f"category:{category}": [row for row in result["rows"] if row["expected"] == category] for category in sorted({row["expected"] for row in rows})})
        summary[name] = {group: summarize(group_rows) for group, group_rows in groups.items() if group_rows}
    paired = {}
    for family in ("expanded_uniform", "scale100", "scale50"):
        rows_by_seed = [summary[f"{family}_{seed}"]["all"] for seed in (7, 17, 27)]
        paired[family] = {
            "seeds": [7, 17, 27],
            "matched_hits_mean": statistics.mean(row["matched_hits"] for row in rows_by_seed),
            "matched_hits_stdev": statistics.stdev(row["matched_hits"] for row in rows_by_seed),
            "expected_class_hits_mean": statistics.mean(row["expected_class_hits"] for row in rows_by_seed),
            "expected_class_hits_stdev": statistics.stdev(row["expected_class_hits"] for row in rows_by_seed),
        }
    baseline = paired["expanded_uniform"]
    comparison = {family: {
        "matched_hits_delta_vs_expanded_uniform": paired[family]["matched_hits_mean"] - baseline["matched_hits_mean"],
        "expected_class_hits_delta_vs_expanded_uniform": paired[family]["expected_class_hits_mean"] - baseline["expected_class_hits_mean"],
    } for family in ("scale100", "scale50")}
    report = {
        "status": "occlusion_shifted_regression_evaluation_complete", "rows": len(rows), "results": results,
        "summary": summary, "paired_summary": paired, "paired_comparison": comparison, "inputs": inputs,
        "training_admitted": False, "promotable": False,
        "limits": ["Fresh development-only holdout; no protected labels or metrics.", "No threshold selection or model promotion."],
    }
    report["identity"] = object_sha256(report); write_json(DATA / "evaluation.json", report)
    print(json.dumps({"paired": paired, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
