"""A* route creation, presentation, and non-fatal quality checks."""

from src.planner.astar_grid import (
    astar,
    cell_to_local_waypoint,
    grid_path_to_local_waypoints,
    simplify_grid_path,
)


def plan_path(planner_config, allow_diagonal):
    grid_path = astar(
        start=planner_config["start"],
        goal=planner_config["goal"],
        obstacles=planner_config["obstacles"],
        width=planner_config["width"],
        height=planner_config["height"],
        allow_diagonal=allow_diagonal,
    )
    simplified_path = simplify_grid_path(grid_path)
    waypoints = grid_path_to_local_waypoints(
        simplified_path,
        resolution_m=planner_config["resolution_m"],
        altitude_m=planner_config["altitude_m"],
        local_frame=planner_config.get("local_frame"),
    )
    return grid_path, simplified_path, waypoints


def print_plan(grid_path, simplified_path, waypoints):
    print(f"Original A* path length: {len(grid_path)}")
    print(f"Simplified waypoint count: {len(waypoints)}")
    print(f"Grid path: {grid_path}")
    print("Generated local NED waypoints:")
    for waypoint in waypoints:
        print(f"  {waypoint}")


def print_coordinate_summary(planner_config):
    gazebo_origin = planner_config.get("gazebo_world_origin_m", [0.0, 0.0, 0.0])
    start_local = cell_to_local_waypoint(
        planner_config["start"], planner_config["resolution_m"], planner_config["altitude_m"], local_frame=planner_config.get("local_frame")
    )
    goal_local = cell_to_local_waypoint(
        planner_config["goal"], planner_config["resolution_m"], planner_config["altitude_m"], local_frame=planner_config.get("local_frame")
    )
    print("\nCoordinate convention:")
    print(f"  Gazebo world xy = map east/north + map origin xy {gazebo_origin[:2]}")
    print(f"  Map-to-local registration: {planner_config.get('local_frame') or 'legacy identity (not calibrated)'}")
    print("  A* grid x = east")
    print("  A* grid y = north")
    print("  MAVSDK local NED: registered horizontal coordinates; down=-route altitude (local datum)")
    print("  Cell-center conversion enabled.")
    print("\nStart:")
    print(
        f"  grid {list(planner_config['start'])} -> "
        f"local east={start_local['east_m']:.3f}, "
        f"north={start_local['north_m']:.3f}, down={start_local['down_m']:.3f}"
    )
    print("\nGoal:")
    print(
        f"  grid {list(planner_config['goal'])} -> "
        f"local east={goal_local['east_m']:.3f}, "
        f"north={goal_local['north_m']:.3f}, down={goal_local['down_m']:.3f}"
    )


def reversed_waypoints(waypoints):
    return [waypoint.copy() for waypoint in reversed(waypoints)]


def cells_adjacent_to(cell, width, height):
    x, y = cell
    adjacent = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                adjacent.add(candidate)
    return adjacent


def validate_path_clearance(label, path, raw_cells, inflated_cells):
    warnings = []
    raw_entries = [cell for cell in path if cell in raw_cells]
    inflated_entries = [cell for cell in path if cell in inflated_cells]
    if raw_entries:
        warnings.append(f"{label} path enters raw physical footprint cells: {raw_entries}")
    if inflated_entries:
        warnings.append(
            f"{label} path enters inflated planning obstacle cells: {inflated_entries}"
        )
    return warnings


def validate_planned_routes(grid_path, return_grid_path, planner_config):
    warnings = []
    start = planner_config["start"]
    goal = planner_config["goal"]
    width = planner_config["width"]
    height = planner_config["height"]
    raw_cells = planner_config["raw_obstacle_cells"]
    inflated_cells = planner_config["inflated_blocking_cells"]
    if start in raw_cells:
        warnings.append(f"start cell {start} is inside a raw physical footprint")
    if goal in raw_cells:
        warnings.append(f"goal cell {goal} is inside a raw physical footprint")
    if start in inflated_cells:
        warnings.append(f"start cell {start} is inside inflated planning obstacle cells")
    if goal in inflated_cells:
        warnings.append(f"goal cell {goal} is inside inflated planning obstacle cells")
    if cells_adjacent_to(goal, width, height) & inflated_cells:
        candidates = []
        for x in range(max(0, goal[0] - 3), min(width, goal[0] + 4)):
            for y in range(max(0, goal[1] - 3), min(height, goal[1] + 4)):
                cell = (x, y)
                if cell in raw_cells or cell in inflated_cells:
                    continue
                if cells_adjacent_to(cell, width, height) & inflated_cells:
                    continue
                candidates.append(cell)
        warnings.append(
            f"goal cell {goal} is adjacent to inflated planning obstacle cells; "
            f"nearby safer candidate goal cells: {candidates[:6]}"
        )
    warnings.extend(validate_path_clearance("Outbound", grid_path, raw_cells, inflated_cells))
    if return_grid_path:
        warnings.extend(
            validate_path_clearance("Return", return_grid_path, raw_cells, inflated_cells)
        )
    warnings.extend(map_quality_warnings(grid_path, planner_config))
    return warnings


def is_border_cell(cell, width, height):
    x, y = cell
    return x == 0 or y == 0 or x == width - 1 or y == height - 1


def nearest_inflated_distance(cell, inflated_cells):
    if not inflated_cells:
        return None
    return min(
        max(abs(cell[0] - obstacle[0]), abs(cell[1] - obstacle[1]))
        for obstacle in inflated_cells
    )


def map_quality_warnings(grid_path, planner_config):
    warnings = []
    width = planner_config["width"]
    height = planner_config["height"]
    inflated_cells = planner_config["inflated_blocking_cells"]
    occupancy_ratio = len(inflated_cells) / (width * height)
    if occupancy_ratio > 0.45:
        warnings.append(
            f"inflated planning obstacles occupy {occupancy_ratio:.1%} of the grid; "
            "internal corridors may be over-blocked"
        )
    if grid_path:
        border_cells = [cell for cell in grid_path if is_border_cell(cell, width, height)]
        border_ratio = len(border_cells) / len(grid_path)
        if border_ratio > 0.35:
            warnings.append(
                f"A* route spends {border_ratio:.1%} of its cells on the outer border; "
                "map may still be forcing a boundary-hugging route"
            )
        internal_cells = [
            cell
            for cell in grid_path
            if 2 <= cell[0] <= width - 3 and 2 <= cell[1] <= height - 3
        ]
        internal_ratio = len(internal_cells) / len(grid_path)
        if internal_ratio < 0.25:
            warnings.append(
                f"A* route only spends {internal_ratio:.1%} of its cells in the internal map area; "
                "no meaningful internal corridor was found"
            )
        distances = [
            nearest_inflated_distance(cell, inflated_cells)
            for cell in grid_path
            if cell not in {planner_config["start"], planner_config["goal"]}
        ]
        distances = [distance for distance in distances if distance is not None]
        if distances and min(distances) < 1:
            warnings.append(
                "A* route has a segment with less than 1 cell clearance from inflated cells"
            )
    return warnings
