"""Compare 50-step and 100-step weights on identical development groups."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.evaluate_stratified_recheck import load_datasets
from scripts.vision.evaluate_scale_stratified_coverage import load_rows as load_scale_rows
from scripts.vision.evaluate_augmentation_ab import load_recovery_rows
from scripts.vision.evaluate_scale_expansion_recheck import score_at_640, summarize


BASE = ROOT / "data/research/ml_training_recovery_v1"
OUT = BASE / "scale-budget-ab-v1"


def main():
    from ultralytics import YOLO

    holdout, dataset_inputs = load_datasets(BASE)
    recovery = load_recovery_rows()
    datasets = {
        "scale_stratified": load_scale_rows(),
        "targeted_coverage": recovery["targeted_coverage"],
        "expansion": recovery["expansion"],
        **holdout,
    }
    weights = {}
    for seed in (7, 17, 27):
        weights[f"steps50_{seed}"] = OUT / f"steps50-{seed}/weights/last.pt"
        weights[f"steps100_{seed}"] = BASE / f"scale-expansion-v1/uniform-{seed}/weights/last.pt"
    inputs = {
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(OUT / "protocol.json"): file_sha256(OUT / "protocol.json"),
        str(BASE / "scale-expansion-v1/protocol.json"): file_sha256(BASE / "scale-expansion-v1/protocol.json"),
        **dataset_inputs,
    }
    for group_rows in datasets.values():
        for row in group_rows:
            inputs[str(Path(row["path"]))] = file_sha256(Path(row["path"]))
    for weight in weights.values():
        inputs[str(weight)] = file_sha256(weight)

    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        groups = {}
        for group, rows in datasets.items():
            groups[group] = [
                {**row, "score": score_at_640(model, Path(row["path"]), row["truth"], row["expected"])}
                for row in rows
            ]
        results[name] = {"weights_sha256": file_sha256(weight), "groups": groups}

    summary = {
        name: {group: summarize(rows) for group, rows in result["groups"].items()}
        for name, result in results.items()
    }
    paired = {}
    for family in ("steps50", "steps100"):
        paired[family] = {}
        for group in datasets:
            rows = [summary[f"{family}_{seed}"][group] for seed in (7, 17, 27)]
            paired[family][group] = {
                "seeds": [7, 17, 27],
                "matched_hits_mean": statistics.mean(row["matched_hits"] for row in rows),
                "matched_hits_stdev": statistics.stdev(row["matched_hits"] for row in rows),
                "expected_class_hits_mean": statistics.mean(row["expected_class_hits"] for row in rows),
                "expected_class_hits_stdev": statistics.stdev(row["expected_class_hits"] for row in rows),
            }

    comparison = {}
    for group in datasets:
        candidate = paired["steps50"][group]
        reference = paired["steps100"][group]
        comparison[group] = {
            "matched_hits_delta_steps50_minus_steps100": candidate["matched_hits_mean"] - reference["matched_hits_mean"],
            "expected_class_hits_delta_steps50_minus_steps100": candidate["expected_class_hits_mean"] - reference["expected_class_hits_mean"],
        }
    rule_groups = ("targeted_coverage", "matrix_controls", "cross_scene")
    comparison["paired_rule_passes_holdout_groups"] = all(
        comparison[group]["matched_hits_delta_steps50_minus_steps100"] >= 0
        and comparison[group]["expected_class_hits_delta_steps50_minus_steps100"] >= 0
        for group in rule_groups
    )

    report = {
        "status": "scale_budget_ab_recheck_complete",
        "protocol_identity": json.loads((OUT / "protocol.json").read_text())["identity"],
        "results": results,
        "summary": summary,
        "paired_budget_summary": paired,
        "paired_comparison": comparison,
        "evaluation_rule_groups": list(rule_groups),
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only comparison on identical data and seed families.",
            "The scale-stratified and expansion groups overlap the training pool; holdout rule uses targeted, controls and cross-scene groups.",
            "No threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(OUT / "recheck.json", report)
    print(json.dumps({"paired": paired, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
