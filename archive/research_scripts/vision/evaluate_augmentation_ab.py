"""Compare small-object augmentation arms on development-only holdout groups."""
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


BASE = ROOT / "data/research/ml_training_recovery_v1"
AB = BASE / "augmentation-ab-v1"
EXPANSION = BASE / "stratified-expansion-v1"
TARGETED = BASE / "targeted-coverage-v1"


def score_at_imgsz(model, path, truth, expected, imgsz):
    """Score one frame with the arm's declared inference size."""
    with Image.open(path) as image:
        result = model.predict(
            image.convert("RGB"), imgsz=imgsz, conf=.37, iou=.7, rect=False,
            device="cpu", verbose=False, agnostic_nms=False, max_det=300,
        )[0]
    preds = [
        {"bbox_xyxy": box, "class_name": model.names[int(cls)], "confidence": conf}
        for box, cls, conf in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        )
    ]
    used = set()
    hits = 0
    for pred in preds:
        overlap, index = max(
            (
                (iou(pred["bbox_xyxy"], obj["bbox_xyxy"]), i)
                for i, obj in enumerate(truth)
                if i not in used and obj["class_name"] == pred["class_name"]
            ),
            default=(0, -1),
        )
        if overlap >= .5:
            used.add(index)
            hits += 1
    expected_hit = expected in TARGETS and any(
        p["class_name"] == expected
        and max(
            (iou(p["bbox_xyxy"], obj["bbox_xyxy"]) for obj in truth if obj["class_name"] == expected),
            default=0,
        ) >= .5
        for p in preds
    )
    return {
        "predictions": len(preds),
        "matched_hits": hits,
        "truth_count": len(truth),
        "expected_class_hit": expected_hit,
        "high_confidence_predictions": sum(float(p["confidence"]) >= .5 for p in preds),
    }


def load_recovery_rows():
    rows = {"targeted_coverage": [], "expansion": []}
    for map_id in ("medium", "complex"):
        receipt_path = TARGETED / map_id / "capture/collection-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        rows["targeted_coverage"].extend(
            {
                "map_id": map_id,
                "view_id": row["view_id"],
                "path": row["rgb_path"],
                "truth": row["truth"]["objects"],
                "expected": row["expected_category"],
            }
            for row in receipt["views"]
        )
    for map_id in ("simple", "medium", "complex"):
        receipt_path = EXPANSION / map_id / "capture/collection-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        rows["expansion"].extend(
            {
                "map_id": map_id,
                "view_id": row["view_id"],
                "path": row["rgb_path"],
                "truth": row["truth"]["objects"],
                "expected": row["expected_category"],
            }
            for row in receipt["views"]
        )
    if len(rows["targeted_coverage"]) != 24 or len(rows["expansion"]) != 36:
        raise ValueError("Unexpected targeted or expansion row count")
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
        "high_confidence_predictions": sum(row["score"]["high_confidence_predictions"] for row in rows),
    }


def main():
    from ultralytics import YOLO

    protocol_path = AB / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    datasets, dataset_inputs = load_datasets(BASE)
    datasets.update(load_recovery_rows())
    inputs = {
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(protocol_path): file_sha256(protocol_path),
        **dataset_inputs,
    }
    for source in (
        TARGETED / "semantic-review.json",
        TARGETED / "dedup-audit.json",
        TARGETED / "manifest.json",
        EXPANSION / "semantic-review.json",
        EXPANSION / "dedup-audit.json",
        EXPANSION / "manifest.json",
        BASE / "expanded-stratified-v1/dataset.yaml",
        BASE / "stratified-sampling-v1/protocol.json",
    ):
        inputs[str(source)] = file_sha256(source)
    for base, maps in ((TARGETED, ("medium", "complex")), (EXPANSION, ("simple", "medium", "complex"))):
        for map_id in maps:
            receipt = base / map_id / "capture/collection-receipt.json"
            inputs[str(receipt)] = file_sha256(receipt)
    for rows in datasets.values():
        for row in rows:
            inputs[str(Path(row["path"]))] = file_sha256(Path(row["path"]))

    weights = {}
    imgsz_by_name = {}
    for arm in ("uniform",):
        for seed in protocol["seeds"]:
            name = f"expanded_{arm}_{seed}"
            weights[name] = BASE / "expanded-stratified-v1" / f"{arm}-{seed}/weights/last.pt"
            imgsz_by_name[name] = 640
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:
            name = f"{arm}_{seed}"
            weights[name] = AB / f"{arm}-{seed}/weights/last.pt"
            imgsz_by_name[name] = protocol["arms"][arm]["imgsz"]
            completion = AB / f"{arm}-{seed}/completion.json"
            inputs[str(completion)] = file_sha256(completion)
    for weight in weights.values():
        if not weight.exists():
            raise FileNotFoundError(weight)
        inputs[str(weight)] = file_sha256(weight)

    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        scored_groups = {}
        for group, rows in datasets.items():
            scored_groups[group] = [
                {
                    **row,
                    "score": score_at_imgsz(
                        model, Path(row["path"]), row["truth"], row["expected"], imgsz_by_name[name]
                    ),
                }
                for row in rows
            ]
        results[name] = {
            "weights_sha256": file_sha256(weight),
            "imgsz": imgsz_by_name[name],
            "groups": scored_groups,
        }

    summary = {
        name: {group: summarize(rows) for group, rows in result["groups"].items()}
        for name, result in results.items()
    }
    paired = {}
    for arm in ("expanded_uniform", "scale_aug", "highres", "cls_weight"):
        paired[arm] = {}
        for group in datasets:
            rows = [summary[f"{arm}_{seed}" if arm != "expanded_uniform" else f"expanded_uniform_{seed}"][group] for seed in protocol["seeds"]]
            paired[arm][group] = {
                "seeds": protocol["seeds"],
                "matched_hits_mean": statistics.mean(row["matched_hits"] for row in rows),
                "matched_hits_stdev": statistics.stdev(row["matched_hits"] for row in rows),
                "expected_class_hits_mean": statistics.mean(row["expected_class_hits"] for row in rows),
                "expected_class_hits_stdev": statistics.stdev(row["expected_class_hits"] for row in rows),
            }

    comparison = {}
    rule_groups = ("targeted_coverage", "matrix_controls", "cross_scene")
    for arm in ("scale_aug", "highres", "cls_weight"):
        comparison[arm] = {}
        for group in datasets:
            baseline = paired["expanded_uniform"][group]
            candidate = paired[arm][group]
            comparison[arm][group] = {
                "matched_hits_delta_vs_expanded_uniform": candidate["matched_hits_mean"] - baseline["matched_hits_mean"],
                "expected_class_hits_delta_vs_expanded_uniform": candidate["expected_class_hits_mean"] - baseline["expected_class_hits_mean"],
            }
        comparison[arm]["paired_rule_passes_holdout_groups"] = all(
            comparison[arm][group]["matched_hits_delta_vs_expanded_uniform"] >= 0
            and comparison[arm][group]["expected_class_hits_delta_vs_expanded_uniform"] >= 0
            for group in rule_groups
        )

    report = {
        "status": "augmentation_ab_recheck_complete",
        "protocol_identity": protocol["identity"],
        "results": results,
        "summary": summary,
        "paired_arm_summary": paired,
        "paired_comparison": comparison,
        "evaluation_rule_groups": list(rule_groups),
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only evaluation; no protected labels or metrics.",
            "The 36-frame expansion group is included for diagnosis but overlaps the training pool and is not a generalization claim.",
            "Targeted coverage, matrix controls, and cross-scene groups are used for the paired holdout rule.",
            "Inference size follows each arm's declared configuration; no threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(AB / "recheck.json", report)
    print(json.dumps({"paired": paired, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
