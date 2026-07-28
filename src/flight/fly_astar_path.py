"""Thin CLI orchestrator for A* planning and PX4 SITL flight execution."""

import os
import sys
from pathlib import Path

from mavsdk import System

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flight.async_runtime import run_with_bounded_shutdown
from src.flight.flight_config import (
    CONNECTION_TIMEOUT_S,
    LANDING_TIMEOUT_S,
    LOGGER_SHUTDOWN_TIMEOUT_S,
    MAX_HORIZONTAL_SPEED_M_S,
    MAX_VERTICAL_SPEED_M_S,
    MIN_RISK_SPEED_M_S,
    OUTPUT_ROOT,
    PLANNER_NAME,
    POSITION_GAIN,
    POSITION_READY_TIMEOUT_S,
    PREVIEW_DIR,
    REACHED_HORIZONTAL_ERROR_M,
    REACHED_VERTICAL_ERROR_M,
    RETURN_SPEED_SCALE,
    TELEMETRY_TIMEOUT_S,
    TURN_SETTLE_S,
    WAYPOINT_TIMEOUT_MODE,
    build_perception_config,
    build_replan_config,
    display_path,
    load_planner_config,
    make_log_path,
    parse_args,
    print_perception_summary,
    print_replan_summary,
    validate_planner_safety,
    validate_runtime_args,
)

MPLCONFIG_DIR = OUTPUT_ROOT / ".matplotlib_cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

from src.flight.landing_manager import (
    attempt_safe_landing,
    configure_landing_timeout,
    wait_until_landed,
)
from src.flight.mavsdk_preflight import (
    health_status_text,
    wait_for_connection as _wait_for_connection,
    wait_for_local_position as _wait_for_local_position,
    wait_for_position_ready as _wait_for_position_ready,
)
from src.flight.mission_lifecycle import execute_flight
from src.flight.perception_response import (
    DangerObstacleDetected,
    build_perception_detector,
    current_perception_detection,
)
from src.flight.preview_writer import save_preview
from src.flight.replanning_controller import (
    attempt_local_replan,
    build_active_replan_route,
    dynamic_cells_from_detection,
    empty_replan_state,
    first_useful_replan_waypoint_index,
    local_position_to_grid_cell,
    replanned_path_to_waypoints,
    reset_replan_event_fields,
    should_attempt_local_replan,
)
from src.flight.route_planning import (
    cells_adjacent_to,
    is_border_cell,
    map_quality_warnings,
    nearest_inflated_distance,
    plan_path,
    print_coordinate_summary,
    print_plan,
    reversed_waypoints,
    validate_planned_routes,
    validate_path_clearance,
)
from src.flight.run_status import status_path_for_log, write_run_status
from src.flight.runtime_settings import FlightRuntimeSettings
from src.flight.telemetry_runtime import (
    configure_logger_timeout,
    log_telemetry,
    watch_armed,
    watch_attitude,
    watch_battery,
    watch_connection_state,
    watch_flight_mode,
    watch_in_air,
    watch_position_velocity_ned,
)
from src.flight.waypoint_executor import (
    clamp,
    configure_runtime,
    fly_astar_waypoints,
    fly_to_waypoint,
    fly_waypoint_route,
    hover_at_waypoint,
    parse_waypoint_timeout,
    print_waypoint_timeout_debug,
    print_waypoint_timeout_info,
    risk_adjusted_speed_scale,
    velocity_command_from_error,
    waypoint_timeout_info,
)
from src.flight.flight_state import (
    ensure_critical_telemetry_fresh,
    horizontal_command_speed,
    horizontal_distance_to_waypoint,
    local_position,
    local_velocity,
    set_phase,
    target_errors,
    telemetry_age_s,
    update_latest,
    value_or_blank,
)

async def wait_for_connection(drone, timeout_s=CONNECTION_TIMEOUT_S):
    return await _wait_for_connection(drone, timeout_s)


async def wait_for_position_ready(drone, timeout_s=POSITION_READY_TIMEOUT_S):
    return await _wait_for_position_ready(drone, timeout_s)


async def wait_for_local_position(latest, timeout_s=TELEMETRY_TIMEOUT_S):
    return await _wait_for_local_position(latest, timeout_s)


def current_runtime_settings():
    return FlightRuntimeSettings(
        max_horizontal_speed_m_s=MAX_HORIZONTAL_SPEED_M_S,
        max_vertical_speed_m_s=MAX_VERTICAL_SPEED_M_S,
        min_risk_speed_m_s=MIN_RISK_SPEED_M_S,
        position_gain=POSITION_GAIN,
        reached_horizontal_error_m=REACHED_HORIZONTAL_ERROR_M,
        reached_vertical_error_m=REACHED_VERTICAL_ERROR_M,
        return_speed_scale=RETURN_SPEED_SCALE,
        turn_settle_s=TURN_SETTLE_S,
        waypoint_timeout_mode=WAYPOINT_TIMEOUT_MODE,
        connection_timeout_s=CONNECTION_TIMEOUT_S,
        position_ready_timeout_s=POSITION_READY_TIMEOUT_S,
        telemetry_timeout_s=TELEMETRY_TIMEOUT_S,
        landing_timeout_s=LANDING_TIMEOUT_S,
        logger_shutdown_timeout_s=LOGGER_SHUTDOWN_TIMEOUT_S,
    )


async def run_flight(
    system_address,
    waypoints,
    planner_info,
    perception_config,
    replan_config,
    perception_detector=None,
    return_home=False,
):
    settings = current_runtime_settings()
    configure_runtime(settings)
    configure_logger_timeout(settings.logger_shutdown_timeout_s)
    configure_landing_timeout(settings.landing_timeout_s)
    services = {
        "system_factory": System,
        "make_log_path": make_log_path,
        "write_run_status": write_run_status,
        "wait_for_connection": wait_for_connection,
        "wait_for_position_ready": wait_for_position_ready,
        "log_telemetry": log_telemetry,
        "fly_astar_waypoints": fly_astar_waypoints,
        "attempt_safe_landing": attempt_safe_landing,
    }
    return await execute_flight(
        system_address,
        waypoints,
        planner_info,
        perception_config,
        replan_config,
        settings,
        services,
        perception_detector,
        return_home,
    )


def main(argv=None):
    args = parse_args(argv)
    global MAX_HORIZONTAL_SPEED_M_S, REACHED_HORIZONTAL_ERROR_M
    global RETURN_SPEED_SCALE, TURN_SETTLE_S
    global WAYPOINT_TIMEOUT_MODE, MIN_RISK_SPEED_M_S
    global CONNECTION_TIMEOUT_S, POSITION_READY_TIMEOUT_S
    global TELEMETRY_TIMEOUT_S, LANDING_TIMEOUT_S, LOGGER_SHUTDOWN_TIMEOUT_S
    validate_runtime_args(args)
    MAX_HORIZONTAL_SPEED_M_S = args.max_speed
    REACHED_HORIZONTAL_ERROR_M = args.waypoint_acceptance
    RETURN_SPEED_SCALE = args.return_speed_scale
    TURN_SETTLE_S = args.turn_settle
    WAYPOINT_TIMEOUT_MODE = parse_waypoint_timeout(args.waypoint_timeout)
    MIN_RISK_SPEED_M_S = args.min_risk_speed
    CONNECTION_TIMEOUT_S = args.connection_timeout
    POSITION_READY_TIMEOUT_S = args.position_ready_timeout
    TELEMETRY_TIMEOUT_S = args.telemetry_timeout
    LANDING_TIMEOUT_S = args.landing_timeout
    LOGGER_SHUTDOWN_TIMEOUT_S = args.logger_shutdown_timeout
    planner_config = load_planner_config(args)
    validate_planner_safety(planner_config)
    perception_config = build_perception_config(args)
    perception_detector = build_perception_detector(args, planner_config)
    replan_config = build_replan_config(args, planner_config)
    grid_path, simplified_path, waypoints = plan_path(planner_config, args.allow_diagonal)
    return_grid_path = list(reversed(grid_path)) if args.return_home else []
    route_warnings = validate_planned_routes(grid_path, return_grid_path, planner_config)
    if route_warnings:
        planner_config["validation_warnings"] = [
            *planner_config.get("validation_warnings", []),
            *route_warnings,
        ]
        print("Route clearance warnings:")
        for warning in route_warnings:
            print(f"  WARNING: {warning}")
    planner_info = {
        "grid_start": str(planner_config["start"]),
        "grid_goal": str(planner_config["goal"]),
        "grid_width": planner_config["width"],
        "grid_height": planner_config["height"],
        "planner_name": PLANNER_NAME,
        "map_name": planner_config["map_name"],
        "resolution_m": planner_config["resolution_m"],
        "altitude_m": planner_config["altitude_m"],
        "return_home_enabled": args.return_home,
    }
    if args.compact_output:
        print(f"Route summary: {len(grid_path)} grid cells, {len(waypoints)} flight waypoints")
        print(
            "Perception: "
            + (f"enabled ({perception_config['risk_action']})" if perception_config["enabled"] else "disabled")
        )
        print(
            "Local replanning: "
            + (f"enabled ({replan_config['mode']})" if replan_config["enabled"] else "disabled")
        )
    else:
        print_coordinate_summary(planner_config)
        print_plan(grid_path, simplified_path, waypoints)
        print_perception_summary(perception_config)
        print_replan_summary(replan_config)
    if args.dry_run:
        print("Dry run requested: not connecting to PX4 and not flying.")
        if args.return_home:
            print("Return-home preview requested.")
        save_preview(
            grid_path,
            simplified_path,
            waypoints,
            args,
            planner_config,
            PREVIEW_DIR,
            PLANNER_NAME,
            display_path,
            reversed_waypoints,
        )
        return
    run_with_bounded_shutdown(
        run_flight(
            args.system_address,
            waypoints,
            planner_info,
            perception_config,
            replan_config,
            perception_detector,
            return_home=args.return_home,
        ),
        LOGGER_SHUTDOWN_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
