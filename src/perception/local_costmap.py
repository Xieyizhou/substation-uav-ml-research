"""Build a rolling body-frame costmap from a 2D laser scan."""

from __future__ import annotations

import math

from src.sensors.types import LaserScanFrame, LocalCostmap


class RollingCostmapBuilder:
    """Preserve unseen recent evidence, decaying it toward unknown over time."""

    def __init__(self, *, decay_half_life_s=1.0, **costmap_options):
        if decay_half_life_s <= 0:
            raise ValueError("decay_half_life_s must be positive")
        self.decay_half_life_s = float(decay_half_life_s)
        self.costmap_options = costmap_options
        self.previous = None
        self.previous_received_s = None
        self.version = 0

    def update(self, scan: LaserScanFrame) -> LocalCostmap:
        current = build_local_costmap(scan, **self.costmap_options)
        occupancy = list(current.occupancy)
        unknown = list(current.unknown)
        elapsed_s = (
            max(0.0, scan.received_monotonic_s - self.previous_received_s)
            if self.previous_received_s is not None
            else 0.0
        )
        if (
            self.previous is not None
            and self.previous.width == current.width
            and self.previous.height == current.height
        ):
            decay = 0.5 ** (elapsed_s / self.decay_half_life_s)
            for index, is_unknown in enumerate(current.unknown):
                if not is_unknown or self.previous.unknown[index]:
                    continue
                decayed = 0.5 + (self.previous.occupancy[index] - 0.5) * decay
                occupancy[index] = decayed
                unknown[index] = abs(decayed - 0.5) < 0.05
        traversability = tuple(
            max(0.0, min(1.0, 1.0 - value if not is_unknown else 0.35))
            for value, is_unknown in zip(occupancy, unknown)
        )
        self.version += 1
        result = LocalCostmap(
            timestamp_s=current.timestamp_s,
            frame_id=current.frame_id,
            resolution_m=current.resolution_m,
            width=current.width,
            height=current.height,
            origin_forward_m=current.origin_forward_m,
            origin_left_m=current.origin_left_m,
            occupancy=tuple(occupancy),
            traversability=traversability,
            unknown=tuple(unknown),
            version=self.version,
        )
        self.previous = result
        self.previous_received_s = scan.received_monotonic_s
        return result


def _cell_for_point(forward_m, left_m, resolution_m, width, height, origin_left_m):
    x = int(math.floor(forward_m / resolution_m))
    y = int(math.floor((left_m - origin_left_m) / resolution_m))
    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def _line_cells(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += step_x
        if twice <= dx:
            error += dx
            y0 += step_y


def build_local_costmap(
    scan: LaserScanFrame,
    *,
    resolution_m=0.25,
    forward_range_m=10.0,
    lateral_range_m=10.0,
    inflation_radius_m=0.5,
) -> LocalCostmap:
    if resolution_m <= 0 or forward_range_m <= 0 or lateral_range_m <= 0:
        raise ValueError("costmap dimensions and resolution must be positive")
    width = int(math.ceil(forward_range_m / resolution_m))
    height = int(math.ceil(lateral_range_m / resolution_m))
    origin_left_m = -lateral_range_m / 2.0
    occupancy = [0.5] * (width * height)
    unknown = [True] * (width * height)
    center_y = int(math.floor((0.0 - origin_left_m) / resolution_m))
    hit_cells = set()

    for index, distance_m in enumerate(scan.ranges_m):
        if not math.isfinite(distance_m) or distance_m < scan.range_min_m:
            continue
        angle = scan.angle_at(index)
        forward_m = distance_m * math.cos(angle)
        left_m = distance_m * math.sin(angle)
        if forward_m < 0:
            continue
        end = _cell_for_point(
            forward_m,
            left_m,
            resolution_m,
            width,
            height,
            origin_left_m,
        )
        if end is None:
            clipped_distance = min(distance_m, forward_range_m)
            end = _cell_for_point(
                clipped_distance * math.cos(angle),
                clipped_distance * math.sin(angle),
                resolution_m,
                width,
                height,
                origin_left_m,
            )
        if end is None:
            continue
        cells = list(_line_cells(0, center_y, *end))
        is_hit = distance_m <= scan.range_max_m
        clear_cells = cells[:-1] if is_hit else cells
        for x, y in clear_cells:
            offset = y * width + x
            occupancy[offset] = 0.0
            unknown[offset] = False
        if is_hit:
            hit_cells.add(end)

    inflation_cells = int(math.ceil(inflation_radius_m / resolution_m))
    for hit_x, hit_y in hit_cells:
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                if dx * dx + dy * dy > inflation_cells * inflation_cells:
                    continue
                x, y = hit_x + dx, hit_y + dy
                if not (0 <= x < width and 0 <= y < height):
                    continue
                offset = y * width + x
                distance_cells = math.hypot(dx, dy)
                occupancy[offset] = max(
                    occupancy[offset],
                    1.0 - 0.45 * distance_cells / max(inflation_cells, 1),
                )
                unknown[offset] = False

    traversability = tuple(
        max(0.0, min(1.0, 1.0 - value if not is_unknown else 0.35))
        for value, is_unknown in zip(occupancy, unknown)
    )
    return LocalCostmap(
        timestamp_s=scan.timestamp_s,
        frame_id=scan.frame_id,
        resolution_m=resolution_m,
        width=width,
        height=height,
        origin_forward_m=0.0,
        origin_left_m=origin_left_m,
        occupancy=tuple(occupancy),
        traversability=traversability,
        unknown=tuple(unknown),
    )
