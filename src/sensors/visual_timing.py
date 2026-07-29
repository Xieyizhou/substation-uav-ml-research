"""Honest summaries for optional visual pipeline stage timings."""

from __future__ import annotations

import math

from src.sensors.types import VisualTiming


VISUAL_DURATION_FIELDS = (
    "capture_to_receive_ms",
    "queue_wait_ms",
    "decode_ms",
    "preprocess_ms",
    "backend_call_ms",
    "inference_ms",
    "postprocess_ms",
    "decision_finalize_ms",
    "end_to_end_ms",
)


def _percentile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_visual_timings(values):
    """Summarize only measured values; null stages never become zero."""
    timings = [
        value if isinstance(value, VisualTiming) else VisualTiming(**value)
        for value in values
    ]
    stages = {}
    for field_name in VISUAL_DURATION_FIELDS:
        measured = [
            float(getattr(timing, field_name))
            for timing in timings
            if getattr(timing, field_name) is not None
        ]
        stages[field_name] = {
            "count": len(measured),
            "p50_ms": _percentile(measured, 0.50),
            "p95_ms": _percentile(measured, 0.95),
            "p99_ms": _percentile(measured, 0.99),
            "min_ms": min(measured) if measured else None,
            "max_ms": max(measured) if measured else None,
            "mean_ms": sum(measured) / len(measured) if measured else None,
        }
    deadline_values = [
        timing.deadline_missed
        for timing in timings
        if timing.deadline_missed is not None
    ]
    return {
        "records": len(timings),
        "stages": stages,
        "deadline_evaluated_count": len(deadline_values),
        "deadline_missed_count": sum(deadline_values),
    }
