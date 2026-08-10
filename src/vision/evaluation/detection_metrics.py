"""Detection matching and frozen confidence-threshold selection."""

from __future__ import annotations

from src.ml import EQUIPMENT_CLASSES


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(truth, predictions, *, threshold, iou_threshold=0.5):
    matched_truth = set()
    matches, false_positives = [], []
    candidates = sorted(
        (item for item in predictions if item["confidence"] >= threshold),
        key=lambda item: -item["confidence"],
    )
    for prediction in candidates:
        choices = [
            (box_iou(prediction["bbox"], item["bbox"]), index)
            for index, item in enumerate(truth)
            if index not in matched_truth
            and item["class_name"] == prediction["class_name"]
        ]
        overlap, index = max(choices, default=(0.0, -1))
        if overlap >= iou_threshold:
            matched_truth.add(index)
            matches.append((index, prediction, overlap))
        else:
            false_positives.append(prediction)
    false_negatives = [
        item for index, item in enumerate(truth) if index not in matched_truth
    ]
    return matches, false_positives, false_negatives


def threshold_metrics(frames, threshold):
    totals = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in EQUIPMENT_CLASSES
    }
    no_target_frames = no_target_false_positives = 0
    small_total = small_matched = 0
    for frame in frames:
        truth = frame["truth"]
        matches, false_positives, false_negatives = match_detections(
            truth, frame["predictions"], threshold=threshold
        )
        matched_indexes = {index for index, _, _ in matches}
        for index, item in enumerate(truth):
            totals[item["class_name"]]["tp" if index in matched_indexes else "fn"] += 1
            if item.get("small"):
                small_total += 1
                small_matched += index in matched_indexes
        for item in false_positives:
            totals[item["class_name"]]["fp"] += 1
        if not truth:
            no_target_frames += 1
            no_target_false_positives += bool(false_positives)
    per_class, f1_values = {}, []
    for name, values in totals.items():
        precision = values["tp"] / max(values["tp"] + values["fp"], 1)
        recall = values["tp"] / max(values["tp"] + values["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {**values, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "threshold": threshold,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "no_target_false_positive_rate": (
            no_target_false_positives / no_target_frames
            if no_target_frames
            else None
        ),
        "small_object_recall": (
            small_matched / small_total if small_total else None
        ),
        "small_object_count": small_total,
    }


def select_confidence_threshold(frames):
    candidates = [
        threshold_metrics(frames, round(value / 100, 2))
        for value in range(5, 76)
    ]
    best = max(candidates, key=lambda row: (row["macro_f1"], -row["threshold"]))
    return {"selected": best, "candidates": candidates}


def onnx_equivalence(pt_frames, onnx_frames):
    if [row["sample_id"] for row in pt_frames] != [
        row["sample_id"] for row in onnx_frames
    ]:
        raise ValueError("PT and ONNX calibration memberships differ")
    total = unmatched = 0
    confidence_differences = []
    for pt, onnx in zip(pt_frames, onnx_frames):
        remaining = list(onnx["predictions"])
        for prediction in pt["predictions"]:
            total += 1
            candidates = [
                (box_iou(prediction["bbox"], item["bbox"]), index, item)
                for index, item in enumerate(remaining)
                if item["class_name"] == prediction["class_name"]
            ]
            overlap, index, match = max(candidates, default=(0.0, -1, None))
            if (
                match is None
                or overlap < 0.99
                or abs(prediction["confidence"] - match["confidence"]) > 0.01
            ):
                unmatched += 1
                continue
            confidence_differences.append(
                abs(prediction["confidence"] - match["confidence"])
            )
            remaining.pop(index)
        unmatched += len(remaining)
        total += len(remaining)
    ratio = unmatched / max(total, 1)
    return {
        "detection_count": total,
        "unmatched_detection_count": unmatched,
        "unmatched_detection_fraction": ratio,
        "maximum_confidence_difference": max(confidence_differences, default=0.0),
        "passed": ratio <= 0.005,
    }
