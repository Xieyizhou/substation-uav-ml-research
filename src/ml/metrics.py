"""Small dependency-free research metrics and latency summaries."""

from __future__ import annotations

import math


def percentile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def classification_report(labels, predictions, classes):
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    per_class = {}
    f1_values = []
    for class_name in classes:
        true_positive = sum(
            label == class_name and prediction == class_name
            for label, prediction in zip(labels, predictions)
        )
        false_positive = sum(
            label != class_name and prediction == class_name
            for label, prediction in zip(labels, predictions)
        )
        false_negative = sum(
            label == class_name and prediction != class_name
            for label, prediction in zip(labels, predictions)
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(label == class_name for label in labels),
        }
    accuracy = (
        sum(label == prediction for label, prediction in zip(labels, predictions))
        / max(len(labels), 1)
    )
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_class": per_class,
    }


def binary_iou(labels, predictions, threshold=0.5):
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    intersection = sum(
        label >= threshold and prediction >= threshold
        for label, prediction in zip(labels, predictions)
    )
    union = sum(
        label >= threshold or prediction >= threshold
        for label, prediction in zip(labels, predictions)
    )
    return intersection / union if union else 1.0


def expected_calibration_error(labels, confidences, predictions, bins=10):
    if not (len(labels) == len(confidences) == len(predictions)):
        raise ValueError("calibration inputs must have equal length")
    total = max(len(labels), 1)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            item
            for item in range(len(labels))
            if low <= confidences[item] < high
            or (index == bins - 1 and confidences[item] == high)
        ]
        if not selected:
            continue
        accuracy = sum(labels[item] == predictions[item] for item in selected) / len(
            selected
        )
        confidence = sum(confidences[item] for item in selected) / len(selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return error


def latency_summary(latencies_ms):
    return {
        "count": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "max_ms": max(latencies_ms) if latencies_ms else None,
    }
