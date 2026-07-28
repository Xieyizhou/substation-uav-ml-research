import json
from pathlib import Path
from typing import Optional

from src.planner.astar_grid import astar

Cell = tuple[int, int]
DEFAULT_ALTITUDE_M = 1.5
DEFAULT_VERTICAL_SAFETY_MARGIN_M = 0.3
DEFAULT_HORIZONTAL_INFLATION_CELLS = 0


def load_obstacle_config(path):
    config_path = Path(path).expanduser()
    with config_path.open() as config_file:
        config = json.load(config_file)

    required_keys = ["width", "height", "start_cell", "goal_cell", "obstacles"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Obstacle config is missing required keys: {missing_keys}")

    return config


def expand_rect_obstacles(config) -> set[Cell]:
    obstacle_cells = set()
    width = int(config["width"])
    height = int(config["height"])

    for obstacle in config.get("obstacles", []):
        obstacle_type = obstacle.get("type")

        if obstacle_type == "rect":
            x_min = int(obstacle["x_min"])
            x_max = int(obstacle["x_max"])
            y_min = int(obstacle["y_min"])
            y_max = int(obstacle["y_max"])

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if 0 <= x < width and 0 <= y < height:
                        obstacle_cells.add((x, y))

        elif obstacle_type == "cell":
            x = int(obstacle["x"])
            y = int(obstacle["y"])
            if 0 <= x < width and 0 <= y < height:
                obstacle_cells.add((x, y))

        else:
            name = obstacle.get("name", "<unnamed>")
            raise ValueError(f"Unsupported obstacle type for {name}: {obstacle_type}")

    return obstacle_cells


def obstacle_blocks_altitude(obstacle, flight_altitude_m, vertical_safety_margin_m):
    z_min_m = float(obstacle.get("z_min_m", float("-inf")))
    z_max_m = float(obstacle.get("z_max_m", float("inf")))
    return (
        flight_altitude_m + vertical_safety_margin_m >= z_min_m
        and flight_altitude_m - vertical_safety_margin_m <= z_max_m
    )


def obstacle_footprint_cells(obstacle, width, height) -> set[Cell]:
    obstacle_type = obstacle.get("type")

    if obstacle_type == "rect":
        cells = {
            (x, y)
            for x in range(int(obstacle["x_min"]), int(obstacle["x_max"]) + 1)
            for y in range(int(obstacle["y_min"]), int(obstacle["y_max"]) + 1)
        }
    elif obstacle_type == "cell":
        cells = {(int(obstacle["x"]), int(obstacle["y"]))}
    else:
        name = obstacle.get("name", "<unnamed>")
        raise ValueError(f"Unsupported obstacle type for {name}: {obstacle_type}")

    return {
        (x, y)
        for x, y in cells
        if 0 <= x < width and 0 <= y < height
    }


def inflate_cells(cells: set[Cell], width, height, inflation_cells) -> set[Cell]:
    inflation_cells = int(inflation_cells)
    if inflation_cells <= 0:
        return set(cells)

    inflated = set()
    for x, y in cells:
        for inflated_x in range(x - inflation_cells, x + inflation_cells + 1):
            for inflated_y in range(y - inflation_cells, y + inflation_cells + 1):
                if 0 <= inflated_x < width and 0 <= inflated_y < height:
                    inflated.add((inflated_x, inflated_y))
    return inflated


def add_cell_names(cell_names: dict[Cell, list[str]], cells: set[Cell], name: str):
    for cell in cells:
        cell_names.setdefault(cell, [])
        if name not in cell_names[cell]:
            cell_names[cell].append(name)


def cells_to_name_map(cell_names: dict[Cell, list[str]]) -> dict[Cell, str]:
    return {cell: ",".join(names) for cell, names in cell_names.items()}


def build_obstacle_map(
    config,
    flight_altitude_m: Optional[float] = None,
    start_cell: Optional[Cell] = None,
    goal_cell: Optional[Cell] = None,
) -> dict[str, object]:
    width = int(config["width"])
    height = int(config["height"])
    if start_cell is None or goal_cell is None:
        config_start, config_goal = get_start_goal(config)
        start_cell = config_start if start_cell is None else start_cell
        goal_cell = config_goal if goal_cell is None else goal_cell

    altitude_m = (
        float(config.get("altitude_m", DEFAULT_ALTITUDE_M))
        if flight_altitude_m is None
        else float(flight_altitude_m)
    )
    vertical_safety_margin_m = float(
        config.get("vertical_safety_margin_m", DEFAULT_VERTICAL_SAFETY_MARGIN_M)
    )
    horizontal_inflation_cells = int(
        config.get("horizontal_inflation_cells", DEFAULT_HORIZONTAL_INFLATION_CELLS)
    )

    raw_obstacle_cells = set()
    raw_blocking_cells = set()
    inflated_blocking_cells = set()
    raw_cell_names: dict[Cell, list[str]] = {}
    raw_blocking_cell_names: dict[Cell, list[str]] = {}
    inflated_cell_names: dict[Cell, list[str]] = {}
    blocking_names = []
    nonblocking_names = []
    warnings = []
    protected_cells = {start_cell, goal_cell}

    for obstacle in config.get("obstacles", []):
        name = obstacle.get("name", "<unnamed>")
        footprint = obstacle_footprint_cells(obstacle, width, height)
        raw_obstacle_cells.update(footprint)
        add_cell_names(raw_cell_names, footprint, name)

        if obstacle_blocks_altitude(obstacle, altitude_m, vertical_safety_margin_m):
            blocking_names.append(name)
            raw_blocking_cells.update(footprint)
            add_cell_names(raw_blocking_cell_names, footprint, name)
            inflated = inflate_cells(footprint, width, height, horizontal_inflation_cells)
            protected_intersections = sorted(inflated & protected_cells)
            if protected_intersections:
                warnings.append(
                    f"obstacle {name} would block protected start/goal cell(s) "
                    f"{protected_intersections} after inflation; keeping them free"
                )
                inflated -= protected_cells
            inflated_blocking_cells.update(inflated)
            add_cell_names(inflated_cell_names, inflated, name)
        else:
            nonblocking_names.append(name)

    inflated_blocking_cells -= protected_cells

    return {
        "altitude_m": altitude_m,
        "vertical_safety_margin_m": vertical_safety_margin_m,
        "horizontal_inflation_cells": horizontal_inflation_cells,
        "raw_obstacle_cells": raw_obstacle_cells,
        "raw_blocking_cells": raw_blocking_cells,
        "inflated_blocking_cells": inflated_blocking_cells,
        "raw_obstacle_cell_to_name": cells_to_name_map(raw_cell_names),
        "raw_blocking_cell_to_name": cells_to_name_map(raw_blocking_cell_names),
        "inflated_obstacle_cell_to_name": cells_to_name_map(inflated_cell_names),
        "blocking_cells": raw_blocking_cells,
        "inflated_cells": inflated_blocking_cells,
        "blocking_obstacle_names": blocking_names,
        "nonblocking_obstacle_names": nonblocking_names,
        "raw_obstacle_cell_count": len(raw_obstacle_cells),
        "raw_blocking_cell_count": len(raw_blocking_cells),
        "obstacle_cell_count": len(raw_blocking_cells),
        "inflated_obstacle_cell_count": len(inflated_blocking_cells),
        "warnings": warnings,
    }


def adjacent_cells(cell: Cell, width: int, height: int) -> set[Cell]:
    x, y = cell
    cells = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            adjacent = (x + dx, y + dy)
            if 0 <= adjacent[0] < width and 0 <= adjacent[1] < height:
                cells.add(adjacent)
    return cells


def nearby_candidate_goal_cells(
    goal: Cell,
    width: int,
    height: int,
    raw_cells: set[Cell],
    inflated_cells: set[Cell],
    limit: int = 6,
) -> list[Cell]:
    candidates = []
    for radius in range(1, max(width, height)):
        for x in range(goal[0] - radius, goal[0] + radius + 1):
            for y in range(goal[1] - radius, goal[1] + radius + 1):
                if max(abs(x - goal[0]), abs(y - goal[1])) != radius:
                    continue
                cell = (x, y)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                if cell in raw_cells or cell in inflated_cells:
                    continue
                if adjacent_cells(cell, width, height) & inflated_cells:
                    continue
                candidates.append(cell)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def get_start_goal(config):
    start = tuple(int(value) for value in config["start_cell"])
    goal = tuple(int(value) for value in config["goal_cell"])
    return start, goal


def get_resolution_altitude(config):
    resolution_m = float(config.get("resolution_m", 1.0))
    altitude_m = float(config.get("altitude_m", DEFAULT_ALTITUDE_M))
    return resolution_m, altitude_m


def validate_obstacle_config(config, allow_diagonal=False, flight_altitude_m=None) -> list[str]:
    warnings = []
    width = int(config["width"])
    height = int(config["height"])
    start, goal = get_start_goal(config)
    obstacle_map = build_obstacle_map(
        config,
        flight_altitude_m=flight_altitude_m,
        start_cell=start,
        goal_cell=goal,
    )
    raw_cells = obstacle_map["raw_obstacle_cells"]
    obstacles = obstacle_map["inflated_blocking_cells"]
    warnings.extend(obstacle_map["warnings"])

    def inside(cell):
        x, y = cell
        return 0 <= x < width and 0 <= y < height

    if not inside(start):
        warnings.append(f"start_cell {start} is outside the {width}x{height} map")
    if not inside(goal):
        warnings.append(f"goal_cell {goal} is outside the {width}x{height} map")
    if start in raw_cells:
        warnings.append(f"start_cell {start} is inside a raw physical footprint")
    if goal in raw_cells:
        warnings.append(f"goal_cell {goal} is inside a raw physical footprint")
    if start in obstacles:
        warnings.append(f"start_cell {start} is inside inflated planning obstacle cells")
    if goal in obstacles:
        warnings.append(f"goal_cell {goal} is inside inflated planning obstacle cells")
    if inside(goal) and adjacent_cells(goal, width, height) & obstacles:
        candidates = nearby_candidate_goal_cells(goal, width, height, raw_cells, obstacles)
        warnings.append(
            f"goal_cell {goal} is adjacent to inflated planning obstacle cells; "
            f"safer nearby candidate goal cells: {candidates}"
        )

    blocked_start_zone = []
    for x in range(min(4, width)):
        for y in range(min(4, height)):
            if (x, y) in obstacles:
                blocked_start_zone.append((x, y))
    if blocked_start_zone:
        warnings.append(
            "nearby start zone cells x=0..3, y=0..3 are not clear: "
            f"{blocked_start_zone}"
        )

    for obstacle in config.get("obstacles", []):
        name = obstacle.get("name", "<unnamed>")
        obstacle_type = obstacle.get("type")
        if obstacle_type == "rect":
            x_min = int(obstacle["x_min"])
            x_max = int(obstacle["x_max"])
            y_min = int(obstacle["y_min"])
            y_max = int(obstacle["y_max"])
            if x_min > x_max or y_min > y_max:
                warnings.append(
                    f"rect obstacle {name} has invalid inclusive bounds: "
                    f"x={x_min}..{x_max}, y={y_min}..{y_max}"
                )
            if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
                warnings.append(
                    f"rect obstacle {name} extends outside the map and will be clipped"
                )
        elif obstacle_type != "cell":
            warnings.append(f"obstacle {name} has unsupported type {obstacle_type}")
        for field in ["z_min_m", "z_max_m"]:
            if field not in obstacle:
                warnings.append(
                    f"obstacle {name} is missing {field}; treating it as vertically unbounded"
                )
        if "z_min_m" in obstacle and "z_max_m" in obstacle:
            z_min_m = float(obstacle["z_min_m"])
            z_max_m = float(obstacle["z_max_m"])
            if z_min_m > z_max_m:
                warnings.append(
                    f"obstacle {name} has invalid vertical bounds: "
                    f"z={z_min_m}..{z_max_m}"
                )

    if inside(start) and inside(goal) and start not in obstacles and goal not in obstacles:
        try:
            astar(
                start=start,
                goal=goal,
                obstacles=obstacles,
                width=width,
                height=height,
                allow_diagonal=allow_diagonal,
            )
        except ValueError as error:
            warnings.append(str(error))

    return warnings
