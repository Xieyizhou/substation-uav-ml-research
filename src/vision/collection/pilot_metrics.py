"""Deterministic source-health and synchronization summaries for visual pilots."""

from __future__ import annotations

from collections import Counter

from src.ml import EQUIPMENT_CLASSES
from src.vision.collection.pilot_acceptance import maximum_consecutive_status


def percentile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    index = min(round((len(values) - 1) * probability), len(values) - 1)
    return values[index]


def distribution(values):
    values = tuple(float(value) for value in values)
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def summarize_source_health(
    frames,
    *,
    expected_source_rate_hz,
    invalid_frame_count=0,
    source_timeout_count=0,
    total_png_bytes=0,
    expected_rate_tolerance_fraction=None,
    jitter_p95_limit_ms=None,
    receive_stall_limit_ms=None,
):
    if expected_source_rate_hz <= 0:
        raise ValueError("expected_source_rate_hz must be positive")
    for name, value in (
        ("expected_rate_tolerance_fraction", expected_rate_tolerance_fraction),
        ("jitter_p95_limit_ms", jitter_p95_limit_ms),
        ("receive_stall_limit_ms", receive_stall_limit_ms),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    frames = tuple(frames)
    timestamps = [frame.capture_timestamp for frame in frames]
    sequences = [frame.sequence_number for frame in frames]
    receive_times = [frame.receive_monotonic_timestamp for frame in frames]
    timestamp_intervals = [
        (current - previous) * 1000.0
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    receive_intervals = [
        (current - previous) * 1000.0
        for previous, current in zip(receive_times, receive_times[1:])
        if current >= previous
    ]
    sequence_counts = Counter(sequences)
    timestamp_counts = Counter(timestamps)
    sequence_gaps = sum(
        max(0, current - previous - 1)
        for previous, current in zip(sequences, sequences[1:])
        if current > previous
    )
    non_monotonic = sum(
        current < previous
        for previous, current in zip(timestamps, timestamps[1:])
    )
    duration = None
    effective_rate = None
    if (
        len(timestamps) >= 2
        and timestamps[-1] > timestamps[0]
        and non_monotonic == 0
    ):
        duration = timestamps[-1] - timestamps[0]
        effective_rate = (len(timestamps) - 1) / duration
    flags = []
    if (
        effective_rate is not None
        and expected_rate_tolerance_fraction is not None
        and expected_source_rate_hz > 0
        and abs(effective_rate - expected_source_rate_hz) / expected_source_rate_hz
        > expected_rate_tolerance_fraction
    ):
        flags.append("observed_rate_outside_configured_tolerance")
    interval_summary = distribution(timestamp_intervals)
    if (
        jitter_p95_limit_ms is not None
        and interval_summary["p95"] is not None
        and interval_summary["p95"] > jitter_p95_limit_ms
    ):
        flags.append("excessive_jitter")
    if sequence_gaps:
        flags.append("sequence_gaps")
    if any(count > 1 for count in timestamp_counts.values()) or non_monotonic:
        flags.append("timestamp_health_failure")
    receive_stalls = (
        sum(value > receive_stall_limit_ms for value in receive_intervals)
        if receive_stall_limit_ms is not None
        else None
    )
    if receive_stalls:
        flags.append("receive_time_stalls")
    return {
        "expected_source_rate_hz": float(expected_source_rate_hz),
        "observed_accepted_frame_count": len(frames),
        "invalid_frame_count": int(invalid_frame_count),
        "first_simulation_timestamp": timestamps[0] if timestamps else None,
        "last_simulation_timestamp": timestamps[-1] if timestamps else None,
        "observed_duration_s": duration,
        "effective_source_rate_hz": effective_rate,
        "inter_frame_interval_ms": interval_summary,
        "sequence_gap_count": sequence_gaps,
        "duplicate_sequence_count": sum(
            count - 1 for count in sequence_counts.values() if count > 1
        ),
        "duplicate_simulation_timestamp_count": sum(
            count - 1 for count in timestamp_counts.values() if count > 1
        ),
        "non_monotonic_simulation_timestamp_count": non_monotonic,
        "receive_inter_frame_interval_ms": distribution(receive_intervals),
        "receive_time_stall_limit_ms": receive_stall_limit_ms,
        "receive_time_stall_count": receive_stalls,
        "source_timeout_count": int(source_timeout_count),
        "total_png_bytes": int(total_png_bytes),
        "observation_flags": sorted(set(flags)),
    }


def summarize_synchronization(synchronized, truths, annotations):
    synchronized = tuple(synchronized)
    truths = tuple(truths)
    annotations = tuple(item for item in annotations if item is not None)
    statuses = Counter(
        item.synchronization.synchronization_status for item in synchronized
    )
    matched_offsets = [
        abs(item.synchronization.synchronization_offset_ms)
        for item in synchronized
        if item.synchronization.synchronization_status
        in {"exact", "nearest_within_tolerance"}
    ]
    object_counts = Counter(
        obj.class_name for annotation in annotations for obj in annotation.objects
    )
    valid_matches = statuses["exact"] + statuses["nearest_within_tolerance"]
    invalid_reasons = Counter(
        reason
        for truth in truths
        if not truth.valid
        for reason in truth.invalid_reasons
    )
    return {
        "rgb_frame_count": len(synchronized),
        "truth_message_count": len(truths),
        "exact_match_count": statuses["exact"],
        "nearest_within_tolerance_count": statuses[
            "nearest_within_tolerance"
        ],
        "unmatched_count": statuses["unmatched"],
        "ambiguous_count": statuses["ambiguous"],
        "invalid_truth_count": sum(not truth.valid for truth in truths),
        "rgb_frames_matched_to_invalid_truth_count": statuses["invalid_truth"],
        "invalid_truth_rgb_fraction": (
            statuses["invalid_truth"] / len(synchronized)
            if synchronized
            else None
        ),
        "maximum_consecutive_invalid_truth_rgb_frames": (
            maximum_consecutive_status(
                [
                    item.synchronization.synchronization_status
                    for item in synchronized
                ],
                "invalid_truth",
            )
        ),
        "invalid_truth_reason_counts": dict(sorted(invalid_reasons.items())),
        "annotation_match_rate": (
            valid_matches / len(synchronized) if synchronized else None
        ),
        "no_target_frame_count": sum(
            item.annotation_status == "verified_no_target" for item in annotations
        ),
        "labelled_target_frame_count": sum(
            item.annotation_status == "labelled" for item in annotations
        ),
        "total_valid_bounding_box_count": sum(
            len(item.objects) for item in annotations
        ),
        "synchronization_absolute_offset_ms": distribution(matched_offsets),
        "per_class_annotation_counts": {
            class_name: object_counts[class_name]
            for class_name in EQUIPMENT_CLASSES
        },
    }
