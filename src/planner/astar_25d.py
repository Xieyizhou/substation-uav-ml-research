"""Deterministic height-layer A* with explicit vertical clearance."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from src.maps.sandbox_geometry import object_cells
from src.planner.obstacle_config import inflate_cells


Node = tuple[int, int, int]


@dataclass(frozen=True)
class HeightLayerGrid:
    width: int
    height: int
    layer_altitudes_m: tuple[float, ...]
    blocked: frozenset[Node]
    horizontal_resolution_m: float = 1.0
    vertical_clearance_m: float = 0.2

    def __post_init__(self):
        if min(self.width, self.height) <= 0:
            raise ValueError("height-layer grid dimensions must be positive")
        if not self.layer_altitudes_m or any(
            second <= first
            for first, second in zip(
                self.layer_altitudes_m, self.layer_altitudes_m[1:]
            )
        ):
            raise ValueError("layer altitudes must be non-empty and increasing")

    def free(self, node):
        east, north, layer = node
        return (
            0 <= east < self.width
            and 0 <= north < self.height
            and 0 <= layer < len(self.layer_altitudes_m)
            and node not in self.blocked
        )


@dataclass(frozen=True)
class HeightLayerRoute:
    nodes: tuple[Node, ...]
    distance_m: float
    horizontal_distance_m: float
    vertical_distance_m: float
    layer_change_count: int


def grid_from_sandbox_map(
    map_value,
    layer_altitudes_m,
    *,
    horizontal_inflation_cells=1,
    drone_half_height_m=0.3,
    vertical_clearance_m=0.2,
):
    layers = tuple(float(value) for value in layer_altitudes_m)
    width, height = int(map_value.width_m), int(map_value.height_m)
    blocked = set()
    for layer, altitude in enumerate(layers):
        raw = set()
        lower_envelope = altitude - drone_half_height_m - vertical_clearance_m
        for item in map_value.objects:
            if lower_envelope <= item.height_m:
                raw.update(object_cells(item, width, height))
        inflated = inflate_cells(
            raw, width, height, int(horizontal_inflation_cells)
        )
        blocked.update((east, north, layer) for east, north in inflated)
    return HeightLayerGrid(
        width, height, layers, frozenset(blocked),
        map_value.resolution_m, vertical_clearance_m,
    )


def _neighbors(grid, node):
    east, north, layer = node
    for de, dn in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        candidate = (east + de, north + dn, layer)
        if grid.free(candidate):
            yield candidate, grid.horizontal_resolution_m
    for delta in (-1, 1):
        next_layer = layer + delta
        candidate = (east, north, next_layer)
        if grid.free(candidate):
            yield candidate, abs(
                grid.layer_altitudes_m[next_layer]
                - grid.layer_altitudes_m[layer]
            )


def _heuristic(grid, node, goal):
    horizontal = (
        abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    ) * grid.horizontal_resolution_m
    vertical = abs(
        grid.layer_altitudes_m[node[2]] - grid.layer_altitudes_m[goal[2]]
    )
    return horizontal + vertical


def plan_height_layer_route(grid, start, goal):
    if not grid.free(start):
        raise ValueError("2.5D start node is outside the grid or blocked")
    if not grid.free(goal):
        raise ValueError("2.5D goal node is outside the grid or blocked")
    queue = [(0.0, 0, start)]
    parent, cost, closed = {}, {start: 0.0}, set()
    sequence = 0
    while queue:
        _, _, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            return _materialize_route(grid, parent, current)
        closed.add(current)
        for candidate, move_cost in _neighbors(grid, current):
            proposed = cost[current] + move_cost
            if proposed >= cost.get(candidate, math.inf):
                continue
            cost[candidate], parent[candidate] = proposed, current
            sequence += 1
            heapq.heappush(
                queue,
                (proposed + _heuristic(grid, candidate, goal), sequence, candidate),
            )
    raise ValueError(f"No 2.5D path found from {start} to {goal}")


def _materialize_route(grid, parent, current):
    nodes = [current]
    while current in parent:
        current = parent[current]
        nodes.append(current)
    nodes.reverse()
    horizontal = vertical = 0.0
    for first, second in zip(nodes, nodes[1:]):
        if first[2] == second[2]:
            horizontal += grid.horizontal_resolution_m
        else:
            vertical += abs(
                grid.layer_altitudes_m[first[2]]
                - grid.layer_altitudes_m[second[2]]
            )
    changes = sum(first[2] != second[2] for first, second in zip(nodes, nodes[1:]))
    return HeightLayerRoute(
        tuple(nodes), round(horizontal + vertical, 6), round(horizontal, 6),
        round(vertical, 6), changes,
    )
