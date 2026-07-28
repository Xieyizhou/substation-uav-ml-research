import heapq
from dataclasses import dataclass
from math import sqrt
from typing import Optional


Cell = tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    obstacles: set[Cell]

    def is_inside(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Cell) -> bool:
        return self.is_inside(cell) and cell not in self.obstacles


def heuristic(cell: Cell, goal: Cell, allow_diagonal: bool) -> float:
    dx = abs(cell[0] - goal[0])
    dy = abs(cell[1] - goal[1])
    if allow_diagonal:
        # Octile distance matches an 8-connected grid with diagonal moves.
        return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
    return dx + dy


def neighbors(cell: Cell, allow_diagonal: bool) -> list[tuple[Cell, float]]:
    x, y = cell
    candidates = [
        ((x + 1, y), 1.0),
        ((x - 1, y), 1.0),
        ((x, y + 1), 1.0),
        ((x, y - 1), 1.0),
    ]

    if allow_diagonal:
        diagonal_cost = sqrt(2)
        candidates.extend(
            [
                ((x + 1, y + 1), diagonal_cost),
                ((x + 1, y - 1), diagonal_cost),
                ((x - 1, y + 1), diagonal_cost),
                ((x - 1, y - 1), diagonal_cost),
            ]
        )

    return candidates


def reconstruct_path(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(
    start: Cell,
    goal: Cell,
    obstacles: set[Cell],
    width: int,
    height: int,
    allow_diagonal: bool = False,
) -> list[Cell]:
    grid = GridMap(width=width, height=height, obstacles=obstacles)

    if not grid.is_free(start):
        raise ValueError(f"Start cell {start} is outside the map or blocked")
    if not grid.is_free(goal):
        raise ValueError(f"Goal cell {goal} is outside the map or blocked")

    open_heap: list[tuple[float, int, Cell]] = []
    tie_breaker = 0
    heapq.heappush(open_heap, (0.0, tie_breaker, start))

    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    closed: set[Cell] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        if current == goal:
            return reconstruct_path(came_from, current)

        closed.add(current)

        for next_cell, move_cost in neighbors(current, allow_diagonal):
            if not grid.is_free(next_cell) or next_cell in closed:
                continue

            tentative_g = g_score[current] + move_cost
            if tentative_g >= g_score.get(next_cell, float("inf")):
                continue

            came_from[next_cell] = current
            g_score[next_cell] = tentative_g
            f_score = tentative_g + heuristic(next_cell, goal, allow_diagonal)
            tie_breaker += 1
            heapq.heappush(open_heap, (f_score, tie_breaker, next_cell))

    raise ValueError(f"No A* path found from {start} to {goal}")


def simplify_grid_path(path: list[Cell]) -> list[Cell]:
    if len(path) <= 2:
        return path[:]

    simplified = [path[0]]
    previous_direction = None

    for index in range(1, len(path)):
        prev_cell = path[index - 1]
        current_cell = path[index]
        direction = (
            current_cell[0] - prev_cell[0],
            current_cell[1] - prev_cell[1],
        )

        if previous_direction is not None and direction != previous_direction:
            simplified.append(prev_cell)

        previous_direction = direction

    simplified.append(path[-1])
    return simplified


def grid_path_to_local_waypoints(
    path: list[Cell],
    resolution_m: float = 1.0,
    altitude_m: float = 2.5,
) -> list[dict[str, object]]:
    waypoints = []
    for index, cell in enumerate(path, start=1):
        waypoints.append(cell_to_local_waypoint(cell, resolution_m, altitude_m, index))
    return waypoints


def cell_to_local_waypoint(
    cell: Cell,
    resolution_m: float = 1.0,
    altitude_m: float = 2.5,
    index: Optional[int] = None,
) -> dict[str, object]:
    x, y = cell
    waypoint = {
        "north_m": (y + 0.5) * resolution_m,
        "east_m": (x + 0.5) * resolution_m,
        "down_m": -altitude_m,
    }
    if index is not None:
        waypoint["name"] = f"WP{index:02d}"
    return waypoint
