"""Evaluate the source-isolated visibility expansion weights.

This is a development-only, fixed-protocol evaluation.  It does not train,
select checkpoints, access sealed labels, or change historical artifacts.
"""
import json
import statistics
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.evaluate_paired_visual_factors import checked_rows
from scripts.vision.evaluate_visual_augmentation_abcd import paired_truth
from scripts.vision.exposure_metrics import score, summary
from scripts.vision.prepare_paired_visual_factors import VARIANTS
from scripts.vision.exposure_protocol import NAMES

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/research/ml_training_recovery_v1"
TRAIN = BASE / "reactor-visibility-expansion-v4/training-control-v1"
PAIR = BASE / "paired-visual-factors-v1"
NEGATIVE = BASE / "hard-negative-isolated-v2/semantic-review.json"
# Versioned output keeps the first report immutable if the reporting layer is
# corrected after inference (the model outputs themselves remain reusable).
OUT = TRAIN / "evaluation-v2"
SEEDS = (7, 17, 27)


def predict(model, path, confidence):
    from PIL import Image
    with Image.open(path) as image:
        result = model.predict(image.convert("RGB"), imgsz=640, conf=confidence,
                               iou=.7, rect=False, device="cpu", verbose=False,
                               agnostic_nms=False, max_det=300)[0]
    return [{"bbox_xyxy": box, "class_name": model.names[int(cls)],
             "confidence": conf}
            for box, cls, conf in zip(result.boxes.xyxy.tolist(),
                                      result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist())]


def _load_inputs():
    review_rows, review_path = checked_rows()
    paired, receipt_inputs = paired_truth(review_rows)
    negative = json.loads(NEGATIVE.read_text())
    if negative.get("status") != "reviewed" or negative.get("accepted") != 48 or negative.get("held"):
        raise ValueError("Negative review incomplete")
    if len(paired) != 48 or len(negative["frames"]) != 48:
        raise ValueError("Development membership incomplete")
    inputs = {str(Path(__file__)): file_sha256(Path(__file__)),
              str(review_path): file_sha256(review_path),
              str(NEGATIVE): file_sha256(NEGATIVE), **receipt_inputs}
    for row in review_rows + negative["frames"]:
        if row.get("decision") != "accepted" or file_sha256(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Review or image hash stale")
        inputs[row["image_path"]] = row["image_sha256"]
    return paired, negative["frames"], inputs


def _weights():
    result = {"v2.11": ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"}
    for seed in SEEDS:
        key = f"visibility-452-{seed}"
        completion = TRAIN / "training" / key / "completion.json"
        if not completion.exists():
            raise ValueError(f"Missing training completion: {key}")
        record = json.loads(completion.read_text())
        if record.get("status") != "trained_not_evaluated" or record.get("optimizer_steps") != 460:
            raise ValueError(f"Training endpoint invalid: {key}")
        weight = Path(record["weights"])
        if not weight.is_file() or file_sha256(weight) != record["weights_sha256"]:
            raise ValueError(f"Training weight hash invalid: {key}")
        if record.get("training_admitted") is not False or record.get("promotable") is not False:
            raise ValueError(f"Forbidden admission flag: {key}")
        result[key] = weight
    return result


def _negative_row(row, model):
    formal = predict(model, row["image_path"], .37)
    low = predict(model, row["image_path"], .001)
    return {"view_id": row["view_id"], "variant": row.get("variant", "no_target"),
            "image_sha256": row["image_sha256"], "predictions": formal,
            "low_predictions": low, "frame_has_prediction": bool(formal)}


def _aggregate(records):
    aggregate = {}
    for variant in VARIANTS:
        vals = [r["summary"][variant] for r in records]
        aggregate[variant] = {
            metric: {"mean": statistics.mean(v[metric] for v in vals),
                     "stdev": statistics.stdev(v[metric] for v in vals),
                     "values": [v[metric] for v in vals]}
            for metric in ("planned_instance_hit_rate", "instance_recall",
                           "matched_precision", "unmatched_predictions")}
        aggregate[variant]["per_class"] = {
            name: {
                metric: {"mean": statistics.mean(v["per_class"][name][metric] for v in vals),
                         "stdev": statistics.stdev(v["per_class"][name][metric] for v in vals),
                         "values": [v["per_class"][name][metric] for v in vals]}
                for metric in ("instance_recall", "matched_precision", "unmatched_predictions")
            } for name in NAMES}
    neg = [r["negative_summary"] for r in records]
    aggregate["no_target"] = {
        metric: {"mean": statistics.mean(v[metric] for v in neg),
                 "stdev": statistics.stdev(v[metric] for v in neg),
                 "values": [v[metric] for v in neg]}
        for metric in ("frame_false_positive_rate", "unmatched_predictions")}
    return aggregate


def _paired_deltas(rows):
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["variant"]] = row
    out = []
    for pair_id, group in sorted(by_pair.items()):
        original = group["original"]
        for variant in VARIANTS[1:]:
            current = group[variant]
            out.append({"pair_id": pair_id, "variant": variant,
                        "category": current["category"],
                        "planned_hit_delta": int(current["planned_instance_hit"]) - int(original["planned_instance_hit"]),
                        "matched_instance_delta": len(current["matches"]) - len(original["matches"]),
                        "unmatched_prediction_delta": current["unmatched_prediction_count"] - original["unmatched_prediction_count"]})
    return out


def _evaluate(name, weight, paired, negatives, inputs):
    path = OUT / f"{name}.json"
    completion = None if name == "v2.11" else TRAIN / "training" / name / "completion.json"
    sources = dict(inputs, **{str(weight): file_sha256(weight)})
    if completion is not None:
        sources[str(completion)] = file_sha256(completion)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("inputs") != sources:
            raise ValueError(f"Cached evaluation inputs changed: {name}")
        return record
    print(f"EVALUATE {name}", flush=True)
    from ultralytics import YOLO
    model = YOLO(str(weight))
    rows = []
    for row, truth in paired:
        rows.append(score(row, truth, predict(model, row["image_path"], .37),
                          predict(model, row["image_path"], .001)))
    neg_rows = [_negative_row(row, model) for row in negatives]
    record = {"status": "complete", "cell": name, "inputs": sources,
              "rows": rows, "negative_rows": neg_rows,
              "paired_deltas": _paired_deltas(rows),
              "summary": {v: summary([r for r in rows if r["variant"] == v]) for v in VARIANTS},
              "negative_summary": {
                  "frames": len(neg_rows),
                  "frames_with_predictions": sum(r["frame_has_prediction"] for r in neg_rows),
                  "frame_false_positive_rate": sum(r["frame_has_prediction"] for r in neg_rows) / len(neg_rows),
                  "unmatched_predictions": sum(len(r["predictions"]) for r in neg_rows)},
              "matching_conflicts": sum(bool(r["matching_conflict"]) for r in rows),
              "runtime_controls": {"device": "cpu", "imgsz": 640, "formal_confidence": .37,
                                   "diagnostic_confidence": .001, "nms_iou": .7,
                                   "max_det": 300, "matching_iou": .5,
                                   "optimizer_created": False, "backward_executed": False,
                                   "training_executed": False, "validation_executed": False},
              "training_admitted": False, "promotable": False,
              "unseen_scene_status": "sealed_not_evaluated"}
    record["identity"] = object_sha256(record)
    write_json(path, record)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired, negatives, inputs = _load_inputs()
    all_weights = _weights()
    records = {}
    for name, weight in all_weights.items():
        records[name] = _evaluate(name, weight, paired, negatives, inputs)
        if len(records[name]["rows"]) != 48 or len(records[name]["negative_rows"]) != 48:
            raise ValueError(f"Evaluation membership incomplete: {name}")
        if records[name]["matching_conflicts"]:
            raise ValueError(f"Matching conflict: {name}")
    trained = [records[f"visibility-452-{s}"] for s in SEEDS]
    aggregate = _aggregate(trained)
    baseline = records["v2.11"]
    baseline_summary = {v: baseline["summary"][v] for v in VARIANTS}
    def difference(a, b):
        # A zero-prediction cell has undefined precision; retain that fact
        # instead of manufacturing a numeric delta.
        return None if a is None or b is None else a - b
    deltas = {}
    for s in SEEDS:
        rec = records[f"visibility-452-{s}"]
        deltas[str(s)] = {
            v: {m: difference(rec["summary"][v][m], baseline_summary[v][m])
                for m in ("planned_instance_hit_rate", "instance_recall",
                          "matched_precision", "unmatched_predictions")}
            for v in VARIANTS
        }
    report = {"status": "development_evaluation_complete_no_admission",
              "weights": {k: str(v) for k, v in all_weights.items()},
              "records": {k: str(OUT / f"{k}.json") for k in records},
              "aggregate_three_seed": aggregate,
              "baseline_v2_11": {"summary": baseline_summary,
                                  "negative_summary": baseline["negative_summary"]},
              "delta_vs_v2_11": deltas,
              "training_fit_scope": "The 388-member training pool is not evaluated here; this report is fixed development inference only.",
              "limits": ["48 paired and 48 no-target images are viewed development data, not blind test data.",
                         "Three source-isolated pilot views per condition do not establish independent-scene generalization.",
                         "Low-threshold outputs remain bounded by NMS and max_det; no qualifying box is not proof of no candidate.",
                         "No threshold tuning, protected-label access, training, admission, promotion, or sealed-scene evaluation."],
              "inputs": {str(Path(__file__)): file_sha256(Path(__file__)),
                         **inputs, **{str(p): file_sha256(p) for p in [OUT / f"{k}.json" for k in records]}},
              "training_admitted": False, "promotable": False,
              "unseen_scene_status": "sealed_not_evaluated"}
    report["identity"] = object_sha256(report)
    write_json(OUT / "summary.json", report)
    print(json.dumps({"status": report["status"], "aggregate": aggregate,
                      "baseline": report["baseline_v2_11"],
                      "delta_vs_v2_11": deltas}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
