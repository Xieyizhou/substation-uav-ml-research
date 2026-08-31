"""Runtime-equivalent class-agnostic YOLO proposal audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, write_json
from src.vision.evaluation.detection_metrics import box_iou
from src.vision.evaluation.yolo_evaluation import collect_predictions


DEFAULT_THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10,
                      0.12, 0.15, 0.20, 0.25)


def _nms(predictions, iou_threshold=0.7, max_proposals=16):
    kept = []
    for row in sorted(predictions, key=lambda item: -item["confidence"]):
        if all(box_iou(row["bbox"], other["bbox"]) < iou_threshold for other in kept):
            kept.append(row)
    return kept[:max_proposals]


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def proposal_metrics(frames, threshold, *, nms_iou=0.7, max_proposals=16):
    class_totals = {name: 0 for name in EQUIPMENT_CLASSES}
    class_matches = {name: 0 for name in EQUIPMENT_CLASSES}
    total = matched = false_positives = small_total = small_matched = 0
    no_target = no_target_with_proposal = 0
    counts = []
    for frame in frames:
        proposals = _nms(
            [row for row in frame["predictions"] if row["confidence"] >= threshold],
            nms_iou, max_proposals,
        )
        counts.append(len(proposals))
        used = set()
        for truth in frame["truth"]:
            name = truth["class_name"]
            total += 1
            class_totals[name] += 1
            if truth.get("small"):
                small_total += 1
            choices = [
                (box_iou(truth["bbox"], row["bbox"]), index)
                for index, row in enumerate(proposals)
                if index not in used
            ]
            overlap, index = max(choices, default=(0.0, -1))
            if overlap >= 0.5:
                used.add(index)
                matched += 1
                class_matches[name] += 1
                small_matched += bool(truth.get("small"))
        false_positives += len(proposals) - len(used)
        if not frame["truth"]:
            no_target += 1
            no_target_with_proposal += bool(proposals)
    per_class = {
        name: {
            "support": class_totals[name],
            "matched": class_matches[name],
            "recall": class_matches[name] / class_totals[name]
            if class_totals[name] else None,
        }
        for name in EQUIPMENT_CLASSES
    }
    present = [row for row in per_class.values() if row["support"]]
    passed = (
        total > 0
        and matched / total >= 0.98
        and all(row["recall"] >= 0.95 for row in present)
        and _percentile(counts, 0.95) <= 16
    )
    return {
        "threshold": float(threshold),
        "nms_iou": float(nms_iou),
        "max_proposals": int(max_proposals),
        "truth_count": total,
        "matched_count": matched,
        "overall_recall": matched / total if total else None,
        "per_class": per_class,
        "small_object_recall": small_matched / small_total if small_total else None,
        "small_object_count": small_total,
        "false_positive_count": false_positives,
        "proposal_count_p50": _percentile(counts, 0.50),
        "proposal_count_p95": _percentile(counts, 0.95),
        "proposal_count_max": max(counts, default=0),
        "no_target_frame_fpr": no_target_with_proposal / no_target if no_target else None,
        "passed": passed,
    }


def audit_proposals(model_path, dataset_root, output_path, *, partition="validation",
                    source_id="unnamed", device="cpu", imgsz=640,
                    thresholds=DEFAULT_THRESHOLDS):
    model_path, dataset_root = Path(model_path), Path(dataset_root)
    frames = collect_predictions(
        model_path, dataset_root, partition, device=device, imgsz=imgsz,
        confidence=0.001, batch=1,
    )
    candidates = [proposal_metrics(frames, value) for value in thresholds]
    passing = [row for row in candidates if row["passed"]]
    selected = max(
        passing,
        key=lambda row: (row["threshold"], -row["false_positive_count"]),
        default=None,
    )
    record = {
        "proposal_audit_schema_version": 1,
        "source_id": str(source_id),
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "dataset": str(dataset_root),
        "partition": partition,
        "frame_count": len(frames),
        "runtime_contract": {
            "batch": 1, "single_label_predictor": True,
            "class_agnostic_nms_iou": 0.7, "max_proposals": 16,
            "input_size": imgsz,
        },
        "thresholds": list(thresholds),
        "candidates": candidates,
        "selected": selected,
        "passed": selected is not None,
        "temporal_track_coverage": None,
        "temporal_unavailable_reason": "dataset lacks an ordered timestamp contract",
    }
    record["identity_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(Path(output_path), record)
    return record
