"""Deterministic flight-duration estimates for visual collection routes."""

from __future__ import annotations

import math


def _distance(points):
    return sum(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(points, points[1:])
    )


def route_timing(route, policy):
    start = (route.start_cell[0] + 0.5, route.start_cell[1] + 0.5)
    outbound = [start]
    for waypoint in route.waypoints:
        cells = [(east + 0.5, north + 0.5) for east, north in waypoint.transit_cells]
        outbound.extend(cells)
        outbound.append((waypoint.east_m, waypoint.north_m))
    returning = [
        outbound[-1],
        *((east + 0.5, north + 0.5) for east, north in route.return_transit_cells),
    ]
    distance_m = _distance(outbound) + _distance(returning)
    movement_waypoint_count = (
        sum(len(waypoint.transit_cells) + 1 for waypoint in route.waypoints)
        + len(route.return_transit_cells)
    )
    expected_s = (
        distance_m / float(policy["nominal_horizontal_speed_m_s"])
        + sum(waypoint.hold_s for waypoint in route.waypoints)
        + movement_waypoint_count * float(policy["waypoint_settle_s"])
        + len(route.waypoints)
        * float(policy["yaw_acquisition_allowance_s_per_observation"])
        + float(policy["fixed_overhead_s"])
    )
    timeout_s = min(
        float(policy["maximum_timeout_s"]),
        max(
            float(policy["minimum_timeout_s"]),
            expected_s * float(policy["timeout_multiplier"]),
        ),
    )
    return {
        "estimated_horizontal_distance_m": distance_m,
        "estimated_movement_waypoint_count": movement_waypoint_count,
        "expected_flight_duration_s": expected_s,
        "flight_timeout_s": timeout_s,
    }
