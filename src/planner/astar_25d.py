"""A* over horizontal cells and discrete altitude layers."""

from __future__ import annotations

import heapq
import math


def _neighbors(node, width, height, layers, allow_diagonal):
    x, y, z = node
    horizontal = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if allow_diagonal:
        horizontal += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in horizontal:
        candidate = (x + dx, y + dy, z)
        if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
            yield candidate, math.hypot(dx, dy)
    for dz in (-1, 1):
        candidate = (x, y, z + dz)
        if 0 <= candidate[2] < layers:
            yield candidate, 1.0


def astar_25d(
    start,
    goal,
    *,
    blocked,
    width,
    height,
    layers,
    climb_cost=2.0,
    allow_diagonal=False,
    cell_costs=None,
):
    if start in blocked or goal in blocked:
        raise ValueError("start and goal must be traversable")
    if any(
        value < 0 or value >= limit
        for value, limit in zip(start, (width, height, layers))
    ) or any(
        value < 0 or value >= limit
        for value, limit in zip(goal, (width, height, layers))
    ):
        raise ValueError("start or goal is outside the 2.5D grid")
    cell_costs = cell_costs or {}
    frontier = [(0.0, start)]
    cost = {start: 0.0}
    parent = {}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return list(reversed(path))
        for candidate, movement_cost in _neighbors(
            current, width, height, layers, allow_diagonal
        ):
            if candidate in blocked:
                continue
            vertical = candidate[2] != current[2]
            candidate_cost = (
                cost[current]
                + movement_cost * (climb_cost if vertical else 1.0)
                + float(cell_costs.get(candidate, 0.0))
            )
            if candidate_cost >= cost.get(candidate, math.inf):
                continue
            cost[candidate] = candidate_cost
            parent[candidate] = current
            heuristic = math.dist(candidate, goal)
            heapq.heappush(frontier, (candidate_cost + heuristic, candidate))
    raise ValueError(f"no 2.5D path found from {start} to {goal}")
