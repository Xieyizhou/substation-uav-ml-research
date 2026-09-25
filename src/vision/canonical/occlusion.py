"""Conservative offline view filtering; never a source of training truth."""

import math


def segment_intersects_bounds(start, end, bounds):
    """Slab intersection for a closed segment and [xmin,xmax,...] box."""
    if len(start) != 3 or len(end) != 3 or len(bounds) != 6:
        raise ValueError("Expected 3D points and six bounds")
    if not all(math.isfinite(x) for x in (*start, *end, *bounds)):
        raise ValueError("Non-finite geometry")
    lo, hi = 0.0, 1.0
    for axis in range(3):
        lower, upper = bounds[2 * axis:2 * axis + 2]
        if lower > upper:
            raise ValueError("Reversed bounds")
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not lower <= start[axis] <= upper:
                return False
            continue
        a, b = sorted(((lower - start[axis]) / delta,
                       (upper - start[axis]) / delta))
        lo, hi = max(lo, a), min(hi, b)
        if lo > hi:
            return False
    return True


def view_blockers(view, objects):
    """Return sorted blockers of the target-center ray, excluding target."""
    targets = [o for o in objects if o['name'] == view['object_id']]
    if len(targets) != 1:
        raise ValueError("View must identify exactly one target")
    bounds = targets[0]['bounds']
    center = [(bounds[i] + bounds[i + 1]) / 2 for i in (0, 2, 4)]
    return sorted(o['name'] for o in objects
                  if o['name'] != view['object_id']
                  and segment_intersects_bounds(view['camera_position'], center,
                                                o['bounds']))
