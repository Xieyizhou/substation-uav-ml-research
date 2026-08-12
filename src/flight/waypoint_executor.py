"""PX4 Offboard waypoint tracking, perception response, and route execution."""

import asyncio
from math import sqrt

from mavsdk.offboard import VelocityNedYaw

from src.flight.flight_config import (
    LANDING_TIMEOUT_S,
    MAX_HORIZONTAL_SPEED_M_S,
    MAX_VERTICAL_SPEED_M_S,
    MIN_RISK_SPEED_M_S,
    POSITION_GAIN,
    REACHED_HORIZONTAL_ERROR_M,
    REACHED_VERTICAL_ERROR_M,
    RETURN_SPEED_SCALE,
    TELEMETRY_TIMEOUT_S,
    TURN_SETTLE_S,
    WAYPOINT_TIMEOUT_MODE,
)
from src.flight.flight_state import (
    ensure_critical_telemetry_fresh,
    horizontal_command_speed,
    horizontal_distance_to_waypoint,
    local_position,
    publish_mission_event,
    set_phase,
    target_errors,
)
from src.flight.landing_manager import wait_until_landed
from src.flight.mavsdk_preflight import wait_for_local_position
from src.flight.perception_response import (
    DangerObstacleDetected,
    SensorDataUnavailable,
    current_perception_detection,
)
from src.flight.replanning_controller import (
    attempt_local_replan,
    build_active_replan_route,
    configure_acceptance,
    route_allows_local_replan,
    update_replanned_route_escape,
    should_attempt_local_replan,
)
from src.flight.route_planning import reversed_waypoints
from src.flight.safety_supervisor import SafetySupervisor
from src.flight.takeoff_stability import (
    takeoff_climb_waypoint,
    validate_takeoff_stability,
)


def configure_runtime(settings):
    global MAX_HORIZONTAL_SPEED_M_S, MAX_VERTICAL_SPEED_M_S, MIN_RISK_SPEED_M_S
    global POSITION_GAIN, REACHED_HORIZONTAL_ERROR_M, REACHED_VERTICAL_ERROR_M
    global RETURN_SPEED_SCALE, TURN_SETTLE_S, TELEMETRY_TIMEOUT_S
    global LANDING_TIMEOUT_S, WAYPOINT_TIMEOUT_MODE
    MAX_HORIZONTAL_SPEED_M_S = settings.max_horizontal_speed_m_s
    MAX_VERTICAL_SPEED_M_S = settings.max_vertical_speed_m_s
    MIN_RISK_SPEED_M_S = settings.min_risk_speed_m_s
    POSITION_GAIN = settings.position_gain
    REACHED_HORIZONTAL_ERROR_M = settings.reached_horizontal_error_m
    REACHED_VERTICAL_ERROR_M = settings.reached_vertical_error_m
    RETURN_SPEED_SCALE = settings.return_speed_scale
    TURN_SETTLE_S = settings.turn_settle_s
    TELEMETRY_TIMEOUT_S = settings.telemetry_timeout_s
    LANDING_TIMEOUT_S = settings.landing_timeout_s
    WAYPOINT_TIMEOUT_MODE = settings.waypoint_timeout_mode
    configure_acceptance(REACHED_HORIZONTAL_ERROR_M, REACHED_VERTICAL_ERROR_M)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def velocity_command_from_error(error, speed_scale=1.0, yaw_deg=0.0):
    north_velocity = POSITION_GAIN * error["north_m"]
    east_velocity = POSITION_GAIN * error["east_m"]
    horizontal_speed = sqrt(north_velocity**2 + east_velocity**2)
    max_horizontal_speed = MAX_HORIZONTAL_SPEED_M_S * speed_scale
    if horizontal_speed > max_horizontal_speed:
        scale = max_horizontal_speed / horizontal_speed
        north_velocity *= scale
        east_velocity *= scale
    down_velocity = clamp(
        POSITION_GAIN * error["down_m"],
        -MAX_VERTICAL_SPEED_M_S,
        MAX_VERTICAL_SPEED_M_S,
    )
    if (
        abs(error["down_m"]) >= REACHED_VERTICAL_ERROR_M
        and abs(down_velocity) < 0.1
    ):
        down_velocity = 0.1 if error["down_m"] > 0.0 else -0.1
    return VelocityNedYaw(
        north_velocity,
        east_velocity,
        down_velocity,
        normalize_yaw_deg(yaw_deg),
    )


def normalize_yaw_deg(yaw_deg):
    normalized = (float(yaw_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def risk_adjusted_speed_scale(base_speed_scale, risk_level, risk_action):
    if risk_action != "slow_down":
        return base_speed_scale
    base_speed_m_s = MAX_HORIZONTAL_SPEED_M_S * base_speed_scale
    if risk_level == "danger":
        adjusted_speed_m_s = max(base_speed_m_s * 0.25, MIN_RISK_SPEED_M_S)
        return adjusted_speed_m_s / MAX_HORIZONTAL_SPEED_M_S
    if risk_level == "warning":
        adjusted_speed_m_s = max(base_speed_m_s * 0.5, MIN_RISK_SPEED_M_S)
        return adjusted_speed_m_s / MAX_HORIZONTAL_SPEED_M_S
    return base_speed_scale


def parse_waypoint_timeout(value):
    if str(value).lower() == "auto":
        return "auto"
    try:
        timeout_s = float(value)
    except ValueError as error:
        raise ValueError("--waypoint-timeout must be 'auto' or a positive number") from error
    if timeout_s <= 0:
        raise ValueError("--waypoint-timeout must be 'auto' or a positive number")
    return timeout_s


def waypoint_timeout_info(position, waypoint, speed_scale, perception_config):
    distance_m = horizontal_distance_to_waypoint(position, waypoint) or 0.0
    expected_speed_m_s = MAX_HORIZONTAL_SPEED_M_S * speed_scale
    if perception_config.get("risk_action", "log_only") == "slow_down":
        expected_speed_m_s = max(expected_speed_m_s * 0.35, MIN_RISK_SPEED_M_S)
    if WAYPOINT_TIMEOUT_MODE == "auto":
        base_timeout_s = distance_m / max(expected_speed_m_s, 0.2)
        timeout_s = max(20.0, base_timeout_s * 3.0 + 10.0)
        if perception_config.get("risk_action", "log_only") == "slow_down":
            timeout_s *= 2.0
    else:
        timeout_s = WAYPOINT_TIMEOUT_MODE
    return {
        "distance_m": distance_m,
        "expected_speed_m_s": expected_speed_m_s,
        "timeout_s": timeout_s,
    }


def print_waypoint_timeout_info(waypoint, timeout_info):
    print(f"Flying to {waypoint['name']}...")
    print(f"  distance: {timeout_info['distance_m']:.2f} m")
    print(f"  expected speed: {timeout_info['expected_speed_m_s']:.2f} m/s")
    print(f"  timeout: {timeout_info['timeout_s']:.1f} s")


def print_waypoint_timeout_debug(
    waypoint, latest, perception_config, perception_detector, command
):
    position = local_position(latest)
    error = target_errors(position, waypoint)
    detection = current_perception_detection(
        perception_config, perception_detector, position, latest["attitude"]
    )
    nearest = detection["nearest_obstacle"] if detection else None
    command_speed = horizontal_command_speed(command)
    print("Waypoint timeout debug:")
    print(f"  waypoint: {waypoint['name']}")
    print(
        "  target N/E/D: "
        f"{waypoint['north_m']:.2f}, {waypoint['east_m']:.2f}, {waypoint['down_m']:.2f}"
    )
    if position is None:
        print("  latest local N/E/D: unavailable")
        print("  current horizontal error: unavailable")
    else:
        print(
            "  latest local N/E/D: "
            f"{position.north_m:.2f}, {position.east_m:.2f}, {position.down_m:.2f}"
        )
        print(f"  current horizontal error: {error['horizontal_m']:.2f} m")
    print(f"  latest perception_risk_level: {detection['risk_level'] if detection else 'clear'}")
    if nearest is None:
        print("  latest nearest obstacle: unavailable")
    else:
        print(
            f"  latest nearest obstacle: {nearest['obstacle_name']} "
            f"at {nearest['distance_m']:.2f} m"
        )
    print(
        "  commanded speed: unavailable"
        if command_speed is None
        else f"  commanded speed: {command_speed:.2f} m/s"
    )


async def fly_to_waypoint(
    drone,
    latest,
    phase_state,
    target_state,
    waypoint,
    phase_name,
    route_direction,
    speed_scale,
    perception_config,
    perception_detector,
    replan_config,
    replan_state,
):
    safety_supervisor = SafetySupervisor(
        stale_after_s=perception_config.get("sensor_stale_after_s", 0.5)
    )
    set_phase(phase_state, phase_name, route_direction)
    target_state.update(waypoint)
    timeout_info = waypoint_timeout_info(
        local_position(latest), waypoint, speed_scale, perception_config
    )
    print_waypoint_timeout_info(waypoint, timeout_info)
    last_command = None
    deadline = asyncio.get_running_loop().time() + timeout_info["timeout_s"]
    while asyncio.get_running_loop().time() < deadline:
        ensure_critical_telemetry_fresh(latest, TELEMETRY_TIMEOUT_S)
        position = local_position(latest)
        if position is None:
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            await asyncio.sleep(0.2)
            continue
        error = target_errors(position, waypoint)
        detection = current_perception_detection(
            perception_config,
            perception_detector,
            position,
            latest["attitude"],
            replan_config=replan_config,
        )
        risk_level = detection["risk_level"] if detection else "clear"
        following_escape_route = update_replanned_route_escape(
            replan_config, replan_state, risk_level
        )
        uses_live_sensor = (
            detection
            and perception_config.get("source", "map_baseline") != "map_baseline"
        )
        safety_decision = (
            safety_supervisor.evaluate(
                detection,
                requested_action=perception_config.get("risk_action", "log_only"),
            )
            if uses_live_sensor
            else None
        )
        if safety_decision and safety_decision.action == "hover_then_land":
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            if detection.get("sensor_healthy") is False:
                raise SensorDataUnavailable(safety_decision.reason)
            raise DangerObstacleDetected(safety_decision.reason)
        if safety_decision and safety_decision.action == "hover":
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            await asyncio.sleep(0.2)
            continue
        now_s = asyncio.get_running_loop().time()
        if route_allows_local_replan(
            replan_config, route_direction
        ) and should_attempt_local_replan(
            replan_config, replan_state, risk_level, now_s
        ):
            replanned_path = attempt_local_replan(
                replan_config, replan_state, position, detection, now_s
            )
            if route_direction == "outbound" and replan_config.get("mode") == "active" and replanned_path:
                replacement_waypoints = build_active_replan_route(
                    replanned_path, replan_config, replan_state, position
                )
                if replacement_waypoints:
                    last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                    await drone.offboard.set_velocity_ned(last_command)
                    return replacement_waypoints
        if (
            safety_decision
            and safety_decision.action == "replan_or_hover"
            and not following_escape_route
            and route_direction != "return"
        ):
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            await asyncio.sleep(0.2)
            continue
        risk_action = perception_config.get("risk_action", "log_only")
        if risk_action == "stop_and_land" and risk_level == "danger":
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            raise DangerObstacleDetected
        if (
            error["horizontal_m"] < REACHED_HORIZONTAL_ERROR_M
            and abs(error["down_m"]) < REACHED_VERTICAL_ERROR_M
        ):
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            print(f"Reached {waypoint['name']}.")
            await asyncio.sleep(TURN_SETTLE_S)
            return None
        adjusted_speed_scale = risk_adjusted_speed_scale(
            speed_scale, risk_level, risk_action
        )
        last_command = velocity_command_from_error(
            error,
            adjusted_speed_scale,
            waypoint.get("yaw_deg", 0.0),
        )
        await drone.offboard.set_velocity_ned(last_command)
        await asyncio.sleep(0.2)
    print_waypoint_timeout_debug(
        waypoint, latest, perception_config, perception_detector, last_command
    )
    raise TimeoutError(f"Timed out before reaching {waypoint['name']}")


async def fly_waypoint_route(
    drone,
    latest,
    phase_state,
    target_state,
    waypoints,
    phase_name,
    route_direction,
    perception_config,
    perception_detector,
    replan_config,
    replan_state,
    speed_scale=1.0,
):
    active_waypoints = list(waypoints)
    completed_waypoints = []
    waypoint_index = 0
    while waypoint_index < len(active_waypoints):
        publish_mission_event(
            phase_state,
            "waypoint_started",
            phase=phase_name,
            route_direction=route_direction,
            waypoint_index=waypoint_index,
            waypoint_count=len(active_waypoints),
            waypoint_name=active_waypoints[waypoint_index]["name"],
            is_final_waypoint=waypoint_index == len(active_waypoints) - 1,
        )
        replacement_waypoints = await fly_to_waypoint(
            drone,
            latest,
            phase_state,
            target_state,
            active_waypoints[waypoint_index],
            phase_name,
            route_direction,
            speed_scale,
            perception_config,
            perception_detector,
            replan_config,
            replan_state,
        )
        if route_direction == "outbound" and replan_config.get("mode") == "active" and replacement_waypoints:
            active_waypoints = list(replacement_waypoints)
            waypoint_index = 0
            print(
                "Continuing outbound flight on active replanned route "
                f"with {len(active_waypoints)} waypoint(s)."
            )
            continue
        completed_waypoints.append(active_waypoints[waypoint_index])
        waypoint_index += 1
    return completed_waypoints


async def hover_at_waypoint(drone, phase_state, target_state, waypoint, phase_name, hover_s):
    set_phase(phase_state, phase_name)
    target_state.update(waypoint)
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(
            0.0,
            0.0,
            0.0,
            normalize_yaw_deg(waypoint.get("yaw_deg", 0.0)),
        )
    )
    await asyncio.sleep(hover_s)


async def fly_astar_waypoints(
    drone,
    latest,
    phase_state,
    target_state,
    waypoints,
    perception_config,
    perception_detector,
    replan_config,
    replan_state,
    return_home=False,
):
    target_takeoff_altitude_m = abs(float(waypoints[0]["down_m"]))
    print(f"Setting PX4 takeoff altitude to {target_takeoff_altitude_m:.2f} m...")
    await drone.action.set_takeoff_altitude(target_takeoff_altitude_m)
    await drone.action.set_takeoff_altitude(target_takeoff_altitude_m)
    set_phase(phase_state, "takeoff")
    print("Arming...")
    await arm_when_ready(drone.action)
    print("Taking off...")
    await drone.action.takeoff()
    print("Waiting 8 seconds for takeoff stabilization...")
    await asyncio.sleep(8)
    await wait_for_local_position(latest, TELEMETRY_TIMEOUT_S)
    validate_takeoff_stability(latest, target_takeoff_altitude_m)
    print("Sending initial zero velocity setpoint before Offboard start...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    print("Starting Offboard mode...")
    await drone.offboard.start()
    print("Offboard mode started.")
    print("Climbing vertically to route altitude...")
    await fly_to_waypoint(
        drone,
        latest,
        phase_state,
        target_state,
        takeoff_climb_waypoint(latest, waypoints[0]["down_m"]),
        "takeoff",
        "none",
        1.0,
        perception_config,
        perception_detector,
        replan_config,
        replan_state,
    )
    print("Flying outbound A* path to goal...")
    completed_outbound_waypoints = await fly_waypoint_route(
        drone,
        latest,
        phase_state,
        target_state,
        waypoints,
        "outbound_to_goal",
        "outbound",
        perception_config,
        perception_detector,
        replan_config,
        replan_state,
        1.0,
    )
    print("Reached goal.")
    print("Hovering briefly at the goal...")
    await hover_at_waypoint(drone, phase_state, target_state, waypoints[-1], "goal_hover", 3)
    if return_home:
        return_waypoints = reversed_waypoints(completed_outbound_waypoints)
        print(
            "Return-home enabled. Flying the verified outbound route "
            "in reverse back to start..."
        )
        print(f"Return route speed scale: {RETURN_SPEED_SCALE:.2f}")
        await fly_waypoint_route(
            drone,
            latest,
            phase_state,
            target_state,
            return_waypoints,
            "return_to_start",
            "return",
            perception_config,
            perception_detector,
            replan_config,
            replan_state,
            RETURN_SPEED_SCALE,
        )
        print("Reached start area.")
        print("Hovering briefly at the start area...")
        await hover_at_waypoint(
            drone, phase_state, target_state, return_waypoints[-1], "start_hover", 2
        )
    else:
        print("Return-home disabled. Landing at goal.")
    set_phase(phase_state, "landing")
    print("Stopping Offboard mode...")
    await drone.offboard.stop()
    print("Offboard mode stopped.")
    print("Landing...")
    await drone.action.land()
    await wait_until_landed(latest, LANDING_TIMEOUT_S)
    set_phase(phase_state, "landed")
from src.flight.arming import arm_when_ready
