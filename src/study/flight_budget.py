"""Route-aware wall-clock budgets for closed-loop study flights."""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path

from src.planner.astar_grid import astar
from src.planner.obstacle_config import build_obstacle_map, get_start_goal


MIN_TIMEOUT_S = 240.0
MAX_TIMEOUT_S = 480.0
FIXED_OVERHEAD_S = 75.0
OUTBOUND_SPEED_M_S = 0.5
RETURN_SPEED_M_S = 0.3
SCHEDULING_MARGIN = 1.35


def flight_timeout_policy(override_s=None):
    """Return the auditable timeout policy bound into formal evidence."""
    if override_s is not None and override_s <= 0:
        raise ValueError("flight timeout must be positive")
    return {
        "fixed_overhead_s": FIXED_OVERHEAD_S,
        "max_timeout_s": MAX_TIMEOUT_S,
        "min_timeout_s": MIN_TIMEOUT_S,
        "outbound_speed_m_s": OUTBOUND_SPEED_M_S,
        "override_s": None if override_s is None else float(override_s),
        "return_speed_m_s": RETURN_SPEED_M_S,
        "scheduling_margin": SCHEDULING_MARGIN,
    }


def route_length_m(planner_path):
    """Return the deterministic A* route length represented by a planner file."""
    config = json.loads(Path(planner_path).read_text(encoding="utf-8"))
    start, goal = get_start_goal(config)
    obstacles = build_obstacle_map(config)["inflated_blocking_cells"]
    path = astar(
        start,
        goal,
        obstacles,
        int(config["width"]),
        int(config["height"]),
        allow_diagonal=bool(config.get("allow_diagonal", False)),
    )
    cells = sum(
        hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(path, path[1:])
    )
    return cells * float(config.get("resolution_m", 1.0))


def closed_loop_timeout_s(planner_path, override_s=None):
    """Choose a bounded timeout, or preserve an explicit operator override."""
    policy = flight_timeout_policy(override_s)
    if policy["override_s"] is not None:
        return policy["override_s"]
    distance_m = route_length_m(planner_path)
    transit_s = distance_m * (
        1.0 / OUTBOUND_SPEED_M_S + 1.0 / RETURN_SPEED_M_S
    )
    estimate = FIXED_OVERHEAD_S + transit_s * SCHEDULING_MARGIN
    return min(MAX_TIMEOUT_S, max(MIN_TIMEOUT_S, estimate))
