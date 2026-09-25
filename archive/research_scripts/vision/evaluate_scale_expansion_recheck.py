"""Recheck the 102-image scale expansion against prior development weights."""
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.evaluate_stratified_recheck import TARGETS, load_datasets
from scripts.vision.evaluate_scale_stratified_coverage import load_rows as load_scale_rows
from scripts.vision.evaluate_augmentation_ab import load_recovery_rows


BASE = ROOT / "data/research/ml_training_recovery_v1"
SCALE = BASE / "scale-expansion-v1"


def score_at_640(model, path, truth, expected):
    with Image.open(path) as image:
        result = model.predict(
            image.convert("RGB"), imgsz=640, conf=.37, iou=.7, rect=False,
            device="cpu", verbose=False, agnostic_nms=False, max_det=300,
        )[0]
    predictions = [
        {"bbox_xyxy": box, "class_name": model.names[int(cls)], "confidence": conf}
        for box, cls, conf in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        )
    ]
    used = set()
    hits = 0
    for prediction in predictions:
        overlap, index = max(
            (
                (iou(prediction["bbox_xyxy"], obj["bbox_xyxy"]), index)
                for index, obj in enumerate(truth)
                if index not in used and obj["class_name"] == prediction["class_name"]
            ),
            default=(0, -1),
        )
        if overlap >= .5:
            used.add(index)
            hits += 1
    expected_hit = expected in TARGETS and any(
        prediction["class_name"] == expected
        and max((iou(prediction["bbox_xyxy"], obj["bbox_xyxy"]) for obj in truth if obj["class_name"] == expected), default=0) >= .5
        for prediction in predictions
    )
    return {
        "predictions": len(predictions),
        "matched_hits": hits,
        "truth_count": len(truth),
        "expected_class_hit": expected_hit,
    }


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

    scale_rows = load_scale_rows()
    recovery_rows = load_recovery_rows()
    holdout, dataset_inputs = load_datasets(BASE)
    datasets = {
        "scale_stratified": scale_rows,
        "targeted_coverage": recovery_rows["targeted_coverage"],
        "expansion": recovery_rows["expansion"],
        **holdout,
    }
    weights = {
        "positive_only": BASE / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": BASE / "negative-ab-v1/fit/weights/last.pt",
    }
    for seed in (7, 17, 27):
        weights[f"expanded_uniform_{seed}"] = BASE / f"expanded-stratified-v1/uniform-{seed}/weights/last.pt"
        weights[f"scale_uniform_{seed}"] = SCALE / f"uniform-{seed}/weights/last.pt"

    inputs = {
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(SCALE / "protocol.json"): file_sha256(SCALE / "protocol.json"),
        str(BASE / "scale-stratified-v1/semantic-review.json"): file_sha256(BASE / "scale-stratified-v1/semantic-review.json"),
        str(BASE / "scale-stratified-v1/dedup-audit.json"): file_sha256(BASE / "scale-stratified-v1/dedup-audit.json"),
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
    for family in ("expanded_uniform", "scale_uniform"):
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

    holdout_groups = ("targeted_coverage", "matrix_controls", "cross_scene")
    comparison = {}
    for group in datasets:
        baseline = paired["expanded_uniform"][group]
        candidate = paired["scale_uniform"][group]
        comparison[group] = {
            "matched_hits_delta_scale_uniform_minus_expanded_uniform": candidate["matched_hits_mean"] - baseline["matched_hits_mean"],
            "expected_class_hits_delta_scale_uniform_minus_expanded_uniform": candidate["expected_class_hits_mean"] - baseline["expected_class_hits_mean"],
        }
    comparison["paired_rule_passes_holdout_groups"] = all(
        comparison[group]["matched_hits_delta_scale_uniform_minus_expanded_uniform"] >= 0
        and comparison[group]["expected_class_hits_delta_scale_uniform_minus_expanded_uniform"] >= 0
        for group in holdout_groups
    )

    report = {
        "status": "scale_expansion_recheck_complete",
        "protocol_identity": json.loads((SCALE / "protocol.json").read_text())["identity"],
        "results": results,
        "summary": summary,
        "paired_arm_summary": paired,
        "paired_comparison": comparison,
        "evaluation_rule_groups": list(holdout_groups),
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only evaluation; no protected labels or metrics.",
            "The scale-stratified and expansion groups overlap the new training pool and are diagnostic fit checks, not generalization claims.",
            "Targeted coverage, matrix controls and cross-scene groups determine the paired rule.",
            "No threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(SCALE / "recheck.json", report)
    print(json.dumps({"paired": paired, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
