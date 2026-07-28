"""Planner map loading, normalization, and safety validation."""

from src.flight.flight_defaults import (
    DEFAULT_MAP_NAME,
    GRID_GOAL,
    GRID_HEIGHT,
    GRID_OBSTACLES,
    GRID_START,
    GRID_WIDTH,
    PROJECT_ROOT,
)
from src.planner.obstacle_config import (
    build_obstacle_map,
    get_resolution_altitude,
    get_start_goal,
    load_obstacle_config,
    validate_obstacle_config,
)


def _list_cells(cells):
    return [[x, y] for x, y in cells]

def validate_planner_safety(planner_config):
    """Reject map configurations that place takeoff or goal in physical obstacles."""
    raw_cells = planner_config["raw_obstacle_cells"]
    errors = []
    if planner_config["start"] in raw_cells:
        errors.append(f"start cell {planner_config['start']} is inside a raw physical footprint")
    if planner_config["goal"] in raw_cells:
        errors.append(f"goal cell {planner_config['goal']} is inside a raw physical footprint")
    if planner_config["altitude_m"] <= 0:
        errors.append("planned flight altitude must be positive")
    if planner_config["resolution_m"] <= 0:
        errors.append("map resolution must be positive")
    if errors:
        raise ValueError("Unsafe planner configuration: " + "; ".join(errors))

def default_planner_config(resolution_m, altitude_m):
    """Build the small built-in grid map used when no obstacle config is passed."""
    if altitude_m is None:
        altitude_m = 2.5
    return {
        "map_name": DEFAULT_MAP_NAME,
        "gazebo_world_origin_m": [0.0, 0.0, 0.0],
        "width": GRID_WIDTH,
        "height": GRID_HEIGHT,
        "start": GRID_START,
        "goal": GRID_GOAL,
        "obstacles": GRID_OBSTACLES,
        "raw_obstacle_cells": GRID_OBSTACLES,
        "raw_blocking_cells": GRID_OBSTACLES,
        "inflated_blocking_cells": GRID_OBSTACLES,
        "blocking_obstacle_cells": GRID_OBSTACLES,
        "inflated_obstacle_cells": GRID_OBSTACLES,
        "raw_obstacle_cell_to_name": {},
        "inflated_obstacle_cell_to_name": {},
        "resolution_m": resolution_m,
        "altitude_m": altitude_m,
        "vertical_safety_margin_m": 0.0,
        "horizontal_inflation_cells": 0,
        "blocking_obstacle_names": [],
        "nonblocking_obstacle_names": [],
        "raw_obstacle_cell_count": len(GRID_OBSTACLES),
        "raw_blocking_cell_count": len(GRID_OBSTACLES),
        "obstacle_cell_count": len(GRID_OBSTACLES),
        "inflated_obstacle_cell_count": len(GRID_OBSTACLES),
        "obstacle_config_path": None,
        "obstacle_config": None,
        "obstacle_map": None,
        "validation_warnings": [],
    }


def load_planner_config(
    args,
    *,
    project_root=PROJECT_ROOT,
    selected_target_resolver,
    display_path_func,
):
    """Load and normalize planner settings from CLI arguments.

    Args:
        args: Parsed CLI arguments from `parse_args`.

    Returns:
        A dictionary containing grid dimensions, start/goal cells, height-aware
        obstacle cells, local waypoint conversion settings, and validation
        warnings. By default this uses the JSON obstacle map paired with the
        current catalog selection. `--obstacle-config` overrides that selection,
        while `--use-built-in-grid` explicitly requests the legacy toy grid.

    Side effects:
        Prints the selected map, obstacle counts, and validation warnings to
        make preview and flight logs self-describing.
    """
    if args.obstacle_config is None:
        print("Using built-in default A* grid.")
        return default_planner_config(args.resolution, args.altitude)

    config_path = args.obstacle_config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_obstacle_config(config_path)
    start, goal = get_start_goal(config)
    selected_target = selected_target_resolver(config_path)
    if selected_target is not None:
        goal = tuple(int(value) for value in selected_target["cell"])
    validation_config = dict(config)
    validation_config["goal_cell"] = list(goal)
    resolution_m, config_altitude_m = get_resolution_altitude(config)
    altitude_m = args.altitude if args.altitude is not None else config_altitude_m
    obstacle_map = build_obstacle_map(
        config,
        flight_altitude_m=altitude_m,
        start_cell=start,
        goal_cell=goal,
    )
    validation_warnings = validate_obstacle_config(
        validation_config,
        args.allow_diagonal,
        flight_altitude_m=altitude_m,
    )
    obstacles = obstacle_map["inflated_blocking_cells"]

    planner_config = {
        "map_name": config.get("map_name", config_path.stem),
        "target_id": selected_target["id"] if selected_target else None,
        "target_display_name": (
            selected_target["display_name"] if selected_target else None
        ),
        "gazebo_world_origin_m": [
            float(value)
            for value in config.get("gazebo_world_origin_m", [0.0, 0.0, 0.0])
        ],
        "width": int(config["width"]),
        "height": int(config["height"]),
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "raw_obstacle_cells": obstacle_map["raw_obstacle_cells"],
        "raw_blocking_cells": obstacle_map["raw_blocking_cells"],
        "inflated_blocking_cells": obstacle_map["inflated_blocking_cells"],
        "blocking_obstacle_cells": obstacle_map["raw_blocking_cells"],
        "inflated_obstacle_cells": obstacle_map["inflated_blocking_cells"],
        "raw_obstacle_cell_to_name": obstacle_map["raw_obstacle_cell_to_name"],
        "inflated_obstacle_cell_to_name": obstacle_map["inflated_obstacle_cell_to_name"],
        "resolution_m": resolution_m,
        "altitude_m": obstacle_map["altitude_m"],
        "vertical_safety_margin_m": obstacle_map["vertical_safety_margin_m"],
        "horizontal_inflation_cells": obstacle_map["horizontal_inflation_cells"],
        "blocking_obstacle_names": obstacle_map["blocking_obstacle_names"],
        "nonblocking_obstacle_names": obstacle_map["nonblocking_obstacle_names"],
        "raw_obstacle_cell_count": obstacle_map["raw_obstacle_cell_count"],
        "raw_blocking_cell_count": obstacle_map["raw_blocking_cell_count"],
        "obstacle_cell_count": obstacle_map["obstacle_cell_count"],
        "inflated_obstacle_cell_count": obstacle_map["inflated_obstacle_cell_count"],
        "obstacle_config_path": config_path,
        "obstacle_config": config,
        "obstacle_map": obstacle_map,
        "validation_warnings": validation_warnings,
    }

    print(f"Using obstacle config: {display_path_func(config_path)}")
    print(f"Map name: {planner_config['map_name']}")
    if selected_target is not None:
        print(
            f"Selected target: {selected_target['id']} — "
            f"{selected_target['display_name']}"
        )
    print(f"Start cell: {planner_config['start']}")
    print(f"Goal cell: {planner_config['goal']}")
    print(f"Resolution: {resolution_m} m/cell")
    print(f"Flight altitude: {planner_config['altitude_m']} m")
    print(
        "Obstacle cells: "
        f"raw={planner_config['raw_obstacle_cell_count']}, "
        f"inflated={planner_config['inflated_obstacle_cell_count']}"
    )
    if not args.compact_output:
        print("Height-aware planning:")
        print(f"  vertical safety margin: {planner_config['vertical_safety_margin_m']} m")
        print(f"  horizontal inflation cells: {planner_config['horizontal_inflation_cells']}")
        print(f"  blocking obstacles: {planner_config['blocking_obstacle_names']}")
        print(
            "  ignored/nonblocking obstacles: "
            f"{planner_config['nonblocking_obstacle_names']}"
        )
        print(
            "Height-blocking raw footprint cells: "
            f"{planner_config['raw_blocking_cell_count']}"
        )
        print(
            "Raw physical footprint cells: "
            f"{_list_cells(sorted(planner_config['raw_obstacle_cells']))}"
        )
        print(
            "Inflated planning obstacle cells: "
            f"{_list_cells(sorted(planner_config['inflated_blocking_cells']))}"
        )
    if validation_warnings:
        print("Obstacle config validation warnings:")
        for warning in validation_warnings:
            print(f"  WARNING: {warning}")
    else:
        print("Obstacle config validation: OK")
    return planner_config

def display_path(path, project_root=PROJECT_ROOT):
    try:
        return path.resolve().relative_to(project_root)
    except ValueError:
        return path.resolve()
