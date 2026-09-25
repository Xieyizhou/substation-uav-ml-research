"""Evaluate the bridge A/B/C/D grid and apply its pre-frozen policy."""
import json
import operator
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.evaluate_paired_visual_factors import checked_rows, match, summarize
from scripts.vision.evaluate_visual_augmentation_abcd import paired_truth, predict
from scripts.vision.prepare_paired_visual_factors import VARIANTS
from scripts.vision.prepare_visual_bridge_training import BASE, OUT, SEEDS, prepare

NEGATIVE_REVIEW = BASE / "hard-negative-isolated-v2/semantic-review.json"


def weights(protocol):
    result = {"frozen_v2_11": Path(protocol["controls"]["initial_weights"])}
    completions = {}
    for arm in "ABCD":
        for seed in SEEDS:
            path = OUT / f"arm-{arm}-seed-{seed}/completion.json"
            row = json.loads(path.read_text())
            weight = Path(row["weights"])
            if row["status"] != "complete" or not row["exposure_verified"] or file_sha256(weight) != row["weights_sha256"]:
                raise ValueError(f"Training cell incomplete: {arm}/{seed}")
            result[f"{arm}_{seed}"] = weight; completions[str(path)] = file_sha256(path)
    return result, completions


def policy_summary(aggregate, arm):
    values = {
        "original.planned_instance_hit_rate.mean": aggregate[arm]["original"]["planned_instance_hit_rate"]["mean"],
        "original.planned_instance_hit_rate.min_seed": aggregate[arm]["original"]["planned_instance_hit_rate"]["min_seed"],
        "material.planned_instance_hit_rate.mean": aggregate[arm]["material"]["planned_instance_hit_rate"]["mean"],
        "material.planned_instance_hit_rate.min_seed": aggregate[arm]["material"]["planned_instance_hit_rate"]["min_seed"],
        "background.planned_instance_hit_rate.mean": aggregate[arm]["background"]["planned_instance_hit_rate"]["mean"],
        "lighting.planned_instance_hit_rate.mean": aggregate[arm]["lighting"]["planned_instance_hit_rate"]["mean"],
        "no_target.frame_false_positive_rate.mean": aggregate[arm]["no_target"]["frame_false_positive_rate"]["mean"],
        "no_target.frame_false_positive_rate.max_seed": aggregate[arm]["no_target"]["frame_false_positive_rate"]["max_seed"],
    }
    operations = {">=": operator.ge, "<=": operator.le}; checks = []
    for rule in prepare()["acceptance_policy"]["required_all"]:
        actual = values[rule["metric"]]
        checks.append({**rule, "actual": actual, "passed": operations[rule["op"]](actual, rule["value"])})
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def main():
    from ultralytics import YOLO
    protocol = prepare(); review_rows, paired_review_path = checked_rows(); paired, receipt_inputs = paired_truth(review_rows)
    negatives = json.loads(NEGATIVE_REVIEW.read_text())
    if negatives["status"] != "reviewed" or negatives["accepted"] != 48 or negatives["held"]:
        raise ValueError("Fixed no-target regression is incomplete")
    weight_paths, completion_inputs = weights(protocol)
    inputs = {str(Path(__file__)): file_sha256(Path(__file__)), str(OUT / "protocol.json"): file_sha256(OUT / "protocol.json"),
              str(paired_review_path): file_sha256(paired_review_path), str(NEGATIVE_REVIEW): file_sha256(NEGATIVE_REVIEW),
              **receipt_inputs, **completion_inputs}
    results = {}
    for name, weight in weight_paths.items():
        print(f"EVALUATE {name}", flush=True); inputs[str(weight)] = file_sha256(weight); model = YOLO(str(weight)); scored = []
        for row, truth in paired:
            predictions = predict(model, row["image_path"]); matches, used, _ = match(predictions, truth); target = row["target_bbox_xyxy"]
            planned = any(pred["class_name"] == row["expected_category"] and iou(pred["bbox_xyxy"], target) >= .5 for pred in predictions)
            scored.append({"pair_id": row["pair_id"], "view_id": row["view_id"], "variant": row["variant"],
                           "category": row["expected_category"], "object_id": row["expected_object_id"],
                           "planned_instance_hit": planned, "truth": truth, "predictions": predictions, "matches": matches,
                           "unmatched_prediction_count": len(predictions) - len(used)})
        negative_rows = []
        for row in negatives["frames"]:
            if row["decision"] != "accepted" or file_sha256(Path(row["image_path"])) != row["image_sha256"]:
                raise ValueError("Fixed negative review changed")
            predictions = predict(model, row["image_path"])
            negative_rows.append({"view_id": row["view_id"], "variant": row["variant"], "predictions": predictions,
                                  "prediction_count": len(predictions), "frame_has_prediction": bool(predictions)})
        summary = {variant: summarize([row for row in scored if row["variant"] == variant]) for variant in VARIANTS}
        negative_summary = {"frames": 48, "frames_with_predictions": sum(row["frame_has_prediction"] for row in negative_rows),
                            "frame_false_positive_rate": sum(row["frame_has_prediction"] for row in negative_rows) / 48,
                            "unmatched_predictions": sum(row["prediction_count"] for row in negative_rows)}
        results[name] = {"weights_path": str(weight), "weights_sha256": file_sha256(weight), "paired_rows": scored,
                         "paired_summary": summary, "negative_rows": negative_rows, "negative_summary": negative_summary}
    aggregate = {}
    for arm in "ABCD":
        aggregate[arm] = {}
        for variant in VARIANTS:
            rows = [results[f"{arm}_{seed}"]["paired_summary"][variant] for seed in SEEDS]
            aggregate[arm][variant] = {}
            for metric in ("planned_instance_hit_rate", "instance_recall", "matched_precision", "unmatched_predictions"):
                values = [row[metric] for row in rows]
                aggregate[arm][variant][metric] = {"mean": statistics.mean(values), "stdev": statistics.stdev(values),
                                                    "min_seed": min(values), "max_seed": max(values), "values": values}
        rows = [results[f"{arm}_{seed}"]["negative_summary"] for seed in SEEDS]; aggregate[arm]["no_target"] = {}
        for metric in ("frame_false_positive_rate", "unmatched_predictions"):
            values = [row[metric] for row in rows]
            aggregate[arm]["no_target"][metric] = {"mean": statistics.mean(values), "stdev": statistics.stdev(values),
                                                    "min_seed": min(values), "max_seed": max(values), "values": values}
    policy = {arm: policy_summary(aggregate, arm) for arm in "ABCD"}
    selected = "D" if policy["D"]["passed"] else "B" if policy["B"]["passed"] else None
    status = "development_candidate_selected" if selected else "development_complete_no_candidate"
    report = {"schema_version": 1, "status": status, "selected_arm": selected,
              "protocol": {"input_size": 640, "device": "cpu", "confidence": .37, "class_aware_nms_iou": .7,
                           "max_det": 300, "match_iou": .5}, "results": results, "aggregate": aggregate,
              "policy_results": policy, "acceptance_policy": protocol["acceptance_policy"], "inputs": inputs,
              "unseen_scene_status": "sealed_not_evaluated", "training_admitted": False, "promotable": False,
              "limits": ["Fixed paired and no-target sets are viewed development regression, not blind tests.",
                         "No threshold tuning, protected-label access, unseen-scene evaluation, or promotion."]}
    report["identity"] = object_sha256(report); write_json(OUT / "development-evaluation.json", report)
    print(json.dumps({"status": status, "selected_arm": selected, "aggregate": aggregate,
                      "policy_results": policy, "identity": report["identity"]}, indent=2))


if __name__ == "__main__":
    main()
