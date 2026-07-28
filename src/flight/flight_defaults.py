"""Shared defaults and filesystem locations for flight execution."""

from pathlib import Path

from src.logging.output_registry import get_previews_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PREVIEW_DIR = get_previews_dir("static_astar") / "as_preview"
DEFAULT_SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
PLANNER_NAME = "astar_grid"
DEFAULT_MAP_NAME = "default_grid"

GRID_WIDTH = 10
GRID_HEIGHT = 10
GRID_START = (0, 0)
GRID_GOAL = (8, 8)
GRID_OBSTACLES = {
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (6, 2),
    (6, 3),
    (6, 4),
    (6, 5),
    (6, 6),
}

MAX_HORIZONTAL_SPEED_M_S = 0.8
MAX_VERTICAL_SPEED_M_S = 0.5
POSITION_GAIN = 0.6
REACHED_HORIZONTAL_ERROR_M = 0.4
REACHED_VERTICAL_ERROR_M = 0.4
WAYPOINT_TIMEOUT_S = 30
WAYPOINT_TIMEOUT_MODE = "auto"
RETURN_SPEED_SCALE = 0.7
TURN_SETTLE_S = 1.0
MIN_RISK_SPEED_M_S = 0.3
CONNECTION_TIMEOUT_S = 30.0
POSITION_READY_TIMEOUT_S = 60.0
TELEMETRY_TIMEOUT_S = 10.0
LANDING_TIMEOUT_S = 45.0
LOGGER_SHUTDOWN_TIMEOUT_S = 10.0
