"""Evaluate expanded-pool weights on expansion, controls and cross-scene sets."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.evaluate_stratified_recheck import load_datasets, score, TARGETS


BASE = ROOT / "data/research/ml_training_recovery_v1"
EXPANSION = BASE / "stratified-expansion-v1"
TRAINING = BASE / "expanded-stratified-v1"


def expansion_rows():
    rows = []
    for map_id in ("simple", "medium", "complex"):
        receipt = json.loads((EXPANSION / map_id / "capture/collection-receipt.json").read_text())
        rows.extend({"path": row["rgb_path"], "truth": row["truth"]["objects"], "expected": row["expected_category"], "view_id": row["view_id"], "map_id": map_id} for row in receipt["views"])
    if len(rows) != 36:
        raise ValueError("Expected 36 expansion frames")
    return rows


def main():
    from ultralytics import YOLO

    datasets, inputs = load_datasets(BASE)
    datasets["expansion"] = expansion_rows()
    inputs[str(EXPANSION / "semantic-review.json")] = file_sha256(EXPANSION / "semantic-review.json")
    inputs[str(EXPANSION / "dedup-audit.json")] = file_sha256(EXPANSION / "dedup-audit.json")
    inputs[str(EXPANSION / "evaluation.json")] = file_sha256(EXPANSION / "evaluation.json")
    inputs[str(TRAINING / "protocol.json")] = file_sha256(TRAINING / "protocol.json")
    inputs[str(Path(__file__))] = file_sha256(Path(__file__))

    weights = {
        "positive_only": BASE / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": BASE / "negative-ab-v1/fit/weights/last.pt",
    }
    protocol = json.loads((TRAINING / "protocol.json").read_text())
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:
            weights[f"expanded-{arm}-{seed}"] = TRAINING / f"{arm}-{seed}/weights/last.pt"

    # The new 66-image pool is disjoint from both existing development groups;
    # compute overlap explicitly rather than relying on view ids.
    training_hashes = {row["image_sha256"] for row in protocol["rows"]}
    overlap = {group: sum(file_sha256(Path(row["path"])) in training_hashes for row in rows) for group, rows in datasets.items()}
    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        groups = {}
        for group, rows in datasets.items():
            scored = []
            for row in rows:
                path = Path(row["path"]); inputs[str(path)] = file_sha256(path)
                scored.append({**row, "score": score(model, path, row["truth"], row["expected"])})
            groups[group] = scored
        results[name] = {"weights_sha256": file_sha256(weight), "groups": groups}

    summary = {}
    for name, result in results.items():
        summary[name] = {}
        for group, rows in result["groups"].items():
            expected = [row for row in rows if row["expected"] in TARGETS]
            summary[name][group] = {
                "frames": len(rows),
                "truth_objects": sum(row["score"]["truth_count"] for row in rows),
                "matched_hits": sum(row["score"]["matched_hits"] for row in rows),
                "expected_class_hits": sum(bool(row["score"]["expected_class_hit"]) for row in expected),
                "expected_class_frames": len(expected),
                "predictions": sum(row["score"]["predictions"] for row in rows),
            }

    paired = {}
    for arm in ("uniform", "stratified"):
        paired[arm] = {}
        for group in datasets:
            values = [summary[f"expanded-{arm}-{seed}"][group] for seed in protocol["seeds"]]
            paired[arm][group] = {
                "matched_hits_mean": statistics.mean(value["matched_hits"] for value in values),
                "matched_hits_stdev": statistics.stdev(value["matched_hits"] for value in values),
                "expected_class_hits_mean": statistics.mean(value["expected_class_hits"] for value in values),
                "expected_class_hits_stdev": statistics.stdev(value["expected_class_hits"] for value in values),
            }
    comparison = {}
    for group in datasets:
        comparison[group] = {
            "matched_hits_delta_stratified_minus_uniform": paired["stratified"][group]["matched_hits_mean"] - paired["uniform"][group]["matched_hits_mean"],
            "expected_class_hits_delta_stratified_minus_uniform": paired["stratified"][group]["expected_class_hits_mean"] - paired["uniform"][group]["expected_class_hits_mean"],
        }

    report = {
        "status": "expanded_recheck_complete",
        "protocol_identity": protocol["identity"],
        "results": results,
        "summary": summary,
        "paired": paired,
        "paired_comparison": comparison,
        "dataset_hash_overlap_with_expanded_training": overlap,
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only simulated evaluation; not protected validation.",
            "Expansion, matrix controls and cross-scene rows are hash-disjoint from the 66-image expanded pool.",
            "No best-seed selection, threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(EXPANSION / "recheck.json", report)
    print(json.dumps({"summary": summary, "paired": paired, "comparison": comparison, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
