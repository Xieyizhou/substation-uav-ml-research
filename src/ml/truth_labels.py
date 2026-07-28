"""Geometry-derived labels for LiDAR risk and traversability training."""

from __future__ import annotations

import math


TRUTH_GENERATOR_VERSION = "lidar-geometry-v2"


def _resample(values, size):
    if not values:
        raise ValueError("scan must contain at least one range")
    if size <= 0:
        raise ValueError("bin count must be positive")
    return [
        values[min(round(index * (len(values) - 1) / max(size - 1, 1)), len(values) - 1)]
        for index in range(size)
    ]


def apply_scan_faults(
    ranges_m,
    *,
    range_max_m,
    noise_stddev_m,
    dropout_probability,
    generator,
):
    """Apply deterministic range noise and missing returns to a scan."""
    output = []
    for distance in ranges_m:
        if generator.random() < dropout_probability:
            output.append(float("inf"))
            continue
        if not math.isfinite(distance):
            output.append(float("inf"))
            continue
        noisy = distance + generator.gauss(0.0, noise_stddev_m)
        output.append(min(max(noisy, 0.0), range_max_m))
    return tuple(output)


def traversability_from_ranges(
    ranges_m,
    *,
    range_max_m,
    bins=72,
    blocked_distance_m=1.5,
    clear_distance_m=5.0,
):
    """Return continuous per-sector traversability in [0, 1]."""
    finite = [
        range_max_m if not math.isfinite(value) else min(max(value, 0.0), range_max_m)
        for value in ranges_m
    ]
    sectors = _resample(finite, bins)
    span = max(clear_distance_m - blocked_distance_m, 1e-6)
    return tuple(
        min(max((distance - blocked_distance_m) / span, 0.0), 1.0)
        for distance in sectors
    )


def collision_time_s(ranges_m, speed_m_s, *, corridor_fraction=0.20):
    if speed_m_s <= 0.05:
        return None
    count = len(ranges_m)
    half = max(1, round(count * corridor_fraction / 2))
    center = count // 2
    corridor = ranges_m[max(0, center - half) : min(count, center + half + 1)]
    finite = [value for value in corridor if math.isfinite(value) and value >= 0]
    return min(finite) / speed_m_s if finite else None


def risk_label_from_truth(
    ranges_m,
    velocity_ned_m_s,
    *,
    braking_deceleration_m_s2=1.8,
    vehicle_radius_m=0.45,
    warning_margin_m=1.0,
):
    speed = math.hypot(float(velocity_ned_m_s[0]), float(velocity_ned_m_s[1]))
    ttc = collision_time_s(ranges_m, speed)
    braking = speed * speed / (2 * max(braking_deceleration_m_s2, 0.1))
    danger_distance = vehicle_radius_m + braking
    warning_distance = danger_distance + warning_margin_m
    finite = [value for value in ranges_m if math.isfinite(value) and value >= 0]
    nearest = min(finite) if finite else float("inf")
    if nearest <= danger_distance or (ttc is not None and ttc <= 1.0):
        return "danger", ttc
    if nearest <= warning_distance or (ttc is not None and ttc <= 2.5):
        return "warning", ttc
    return "clear", ttc


def recommended_direction_deg(traversability, *, max_angle_deg=90.0):
    """Select the safest sector, preferring smaller turns when scores tie."""
    if not traversability:
        raise ValueError("traversability must not be empty")
    denominator = max(len(traversability) - 1, 1)
    angles = [
        -max_angle_deg + (2 * max_angle_deg * index / denominator)
        for index in range(len(traversability))
    ]
    best = max(
        range(len(traversability)),
        key=lambda index: (float(traversability[index]), -abs(angles[index])),
    )
    return angles[best]


def occupancy_from_traversability(traversability):
    return tuple(1.0 - float(value) for value in traversability)


def label_scan(ranges_m, range_max_m, velocity_ned_m_s, *, bins=72):
    traversability = traversability_from_ranges(
        ranges_m, range_max_m=range_max_m, bins=bins
    )
    risk, ttc = risk_label_from_truth(ranges_m, velocity_ned_m_s)
    return {
        "risk_label": risk,
        "safety_label": risk,
        "traversability": traversability,
        "recommended_direction_deg": recommended_direction_deg(traversability),
        "ground_truth_occupancy": occupancy_from_traversability(traversability),
        "collision_time_s": ttc,
    }
