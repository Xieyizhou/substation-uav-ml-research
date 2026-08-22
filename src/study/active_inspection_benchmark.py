"""Deterministic offline baselines for active sandbox inspection."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.planner.active_inspection import RuntimeMap, _astar


def _route_distance(grid, start, goals):
    current, distance, reachable = start, 0.0, True
    for goal in goals:
        path = _astar(grid, current, goal)
        if not path:
            reachable = False
            continue
        distance += (len(path) - 1) * grid.resolution_m
        current = goal
    return distance, reachable


def _track_viewpoints(grid, tracks):
    points = []
    for track in tracks:
        center = round(track["east_m"] / grid.resolution_m), round(track["north_m"] / grid.resolution_m)
        radius = max(1, round(5 / grid.resolution_m))
        candidates = []
        for degrees in range(0, 360, 45):
            angle = math.radians(degrees)
            cell = round(center[0] + radius * math.cos(angle)), round(center[1] + radius * math.sin(angle))
            if 0 <= cell[0] < grid.width_cells and 0 <= cell[1] < grid.height_cells and cell not in grid.occupied_cells:
                candidates.append(cell)
        if candidates:
            points.append((track["target_id"], min(candidates, key=lambda c: (c[1], c[0]))))
    return points


def compare_schedulers(plan: Mapping[str, Any], map_value: Mapping[str, Any]) -> dict[str, Any]:
    grid = RuntimeMap.from_mapping(map_value)
    start = (
        round(float(map_value.get("start_east_m", 0)) / grid.resolution_m),
        round(float(map_value.get("start_north_m", 0)) / grid.resolution_m),
    )
    stride = 4
    exploration = []
    for row, y in enumerate(range(1, grid.height_cells - 1, stride)):
        xs = list(range(1, grid.width_cells - 1, stride))
        if row % 2:
            xs.reverse()
        exploration.extend((x, y) for x in xs if (x, y) not in grid.occupied_cells)
    fixed_distance, fixed_reachable = _route_distance(grid, start, exploration)
    remaining = _track_viewpoints(grid, [t for t in plan.get("tracks", []) if not t.get("ambiguous")])
    nearest_order, current = [], start
    while remaining:
        selected = min(remaining, key=lambda item: (abs(item[1][0] - current[0]) + abs(item[1][1] - current[1]), item[0]))
        nearest_order.append(selected[1])
        current = selected[1]
        remaining.remove(selected)
    nearest_distance, nearest_reachable = _route_distance(grid, start, nearest_order + exploration)
    active_cells = [cell for decision in plan.get("decisions", []) for cell in decision.get("transit_cells", [])[1:]]
    active_distance = len(active_cells) * grid.resolution_m
    collision_count = sum(tuple(cell) in grid.occupied_cells for cell in active_cells)
    improvement = (fixed_distance - active_distance) / fixed_distance if fixed_distance else 0
    tracks = [t for t in plan.get("tracks", []) if not t.get("ambiguous")]
    mission_complete = plan.get("exploration_coverage", 0) >= .90 and all(t.get("inspected") for t in tracks)
    return {
        "schema_version": 1,
        "baselines": {
            "fixed_serpentine": {"distance_m": round(fixed_distance, 8), "reachable": fixed_reachable},
            "nearest_target_first": {"distance_m": round(nearest_distance, 8), "reachable": nearest_reachable},
            "active_utility": {"distance_m": round(active_distance, 8), "reachable": collision_count == 0},
        },
        "active_improvement_over_fixed": round(improvement, 8),
        "collision_count": collision_count,
        "safety_gate_failures": collision_count,
        "mission_complete": mission_complete,
    }


def acceptance(report: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    recall = float(report.get("target_recall", 0))
    accuracy = float(report.get("class_accuracy", 0))
    checks = {
        "target_recall": recall >= .95,
        "class_accuracy": accuracy >= .90,
        "zero_collision": benchmark["collision_count"] == 0,
        "zero_safety_gate_failures": benchmark["safety_gate_failures"] == 0,
        "mission_complete": bool(benchmark["mission_complete"]),
        "efficiency_improvement": benchmark["active_improvement_over_fixed"] >= .15,
    }
    return {"passed": all(checks.values()), "checks": checks, "evidence_level": "deterministic_offline"}
