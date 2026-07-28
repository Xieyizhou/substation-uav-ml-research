"""Dependency-free point-cloud to BEV height, occupancy, and clearance grids."""

from __future__ import annotations

import math


def point_cloud_to_bev(
    points_xyz_m,
    *,
    forward_range_m=20.0,
    lateral_range_m=20.0,
    height_range_m=(-3.0, 5.0),
    resolution_m=0.5,
    ground_threshold_m=0.2,
):
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    width = int(math.ceil(forward_range_m / resolution_m))
    height = int(math.ceil(lateral_range_m / resolution_m))
    minimum = [math.inf] * (width * height)
    maximum = [-math.inf] * (width * height)
    counts = [0] * (width * height)
    lateral_origin = -lateral_range_m / 2.0
    for forward, left, up in points_xyz_m:
        if not all(math.isfinite(value) for value in (forward, left, up)):
            continue
        if not (0 <= forward < forward_range_m):
            continue
        if not (lateral_origin <= left < -lateral_origin):
            continue
        if not (height_range_m[0] <= up <= height_range_m[1]):
            continue
        x = int(forward / resolution_m)
        y = int((left - lateral_origin) / resolution_m)
        offset = y * width + x
        minimum[offset] = min(minimum[offset], up)
        maximum[offset] = max(maximum[offset], up)
        counts[offset] += 1
    occupancy = tuple(count > 0 and maximum[index] > ground_threshold_m for index, count in enumerate(counts))
    height_span = tuple(
        maximum[index] - minimum[index] if count else 0.0
        for index, count in enumerate(counts)
    )
    clearance = tuple(
        max(0.0, height_range_m[1] - maximum[index]) if count else height_range_m[1]
        for index, count in enumerate(counts)
    )
    return {
        "width": width,
        "height": height,
        "resolution_m": resolution_m,
        "lateral_origin_m": lateral_origin,
        "occupancy": occupancy,
        "height_span_m": height_span,
        "clearance_m": clearance,
        "point_count": tuple(counts),
    }
