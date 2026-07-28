"""Backward-compatible flight configuration exports."""

from datetime import datetime, timezone

from src.flight.flight_cli import build_argument_parser
from src.flight.flight_defaults import (
    CONNECTION_TIMEOUT_S,
    DEFAULT_MAP_NAME,
    DEFAULT_SYSTEM_ADDRESS,
    GRID_GOAL,
    GRID_HEIGHT,
    GRID_OBSTACLES,
    GRID_START,
    GRID_WIDTH,
    LANDING_TIMEOUT_S,
    LOGGER_SHUTDOWN_TIMEOUT_S,
    LOG_DIR,
    MAX_HORIZONTAL_SPEED_M_S,
    MAX_VERTICAL_SPEED_M_S,
    MIN_RISK_SPEED_M_S,
    OUTPUT_ROOT,
    PLANNER_NAME,
    POSITION_GAIN,
    POSITION_READY_TIMEOUT_S,
    PREVIEW_DIR,
    PROJECT_ROOT,
    REACHED_HORIZONTAL_ERROR_M,
    REACHED_VERTICAL_ERROR_M,
    RETURN_SPEED_SCALE,
    TELEMETRY_TIMEOUT_S,
    TURN_SETTLE_S,
    WAYPOINT_TIMEOUT_MODE,
    WAYPOINT_TIMEOUT_S,
)
from src.flight.flight_planner_config import (
    _list_cells,
    default_planner_config,
    display_path as _display_path,
    load_planner_config as _load_planner_config,
    validate_planner_safety,
)
from src.flight.flight_runtime_config import (
    build_perception_config,
    build_replan_config,
    print_perception_summary,
    print_replan_summary,
    validate_runtime_args,
)


def selected_obstacle_config_path():
    """Return the obstacle config paired with the currently selected Gazebo map."""
    from src.maps.map_catalog import current_map, project_path

    return project_path(current_map()["obstacle_config"])


def selected_target_for_config_path(config_path):
    """Return the saved target preset when a config belongs to the map catalog."""
    from src.maps.target_catalog import selected_target_for_config

    return selected_target_for_config(config_path)


def parse_args(argv=None):
    """Parse flight arguments and resolve the selected map when needed."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.obstacle_config is None and not args.use_built_in_grid:
        args.obstacle_config = selected_obstacle_config_path()
    return args


def load_planner_config(args):
    """Load planner settings while preserving patchable compatibility hooks."""
    return _load_planner_config(
        args,
        project_root=PROJECT_ROOT,
        selected_target_resolver=selected_target_for_config_path,
        display_path_func=display_path,
    )


def display_path(path):
    return _display_path(path, project_root=PROJECT_ROOT)


def make_log_path():
    """Return the timestamped CSV path for the next A* telemetry log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"astar_{timestamp}.csv"
