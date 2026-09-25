"""PX4 Offboard waypoint tracking, perception response, and route execution."""

import asyncio
from math import sqrt

from mavsdk.offboard import VelocityNedYaw

from src.flight.active_replan_runtime import (
    active_replan_replacement,
    publish_dynamic_replan_event,
)
from src.flight.active_semantic_runtime import active_semantic_replacement, complete_semantic_waypoint
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
    finish_or_raise_waypoint_timeout,
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
    configure_acceptance,
    update_replanned_route_escape,
)
from src.flight.route_planning import reversed_waypoints
from src.flight.safety_supervisor import SafetySupervisor
from src.flight.takeoff_stability import (
    takeoff_climb_waypoint,
    validate_takeoff_stability,
    wait_for_takeoff_hover,
)
from src.flight.waypoint_progress import WaypointProgressWatchdog
from src.flight.waypoint_policy import (
    clamp, normalize_yaw_deg, return_has_geometric_obstacle_evidence,
    print_waypoint_timeout_info, print_timeout_debug, timeout_info,
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


def waypoint_dwell_s(waypoint):
    requested = float(waypoint.get("dwell_s", 0.0))
    return max(TURN_SETTLE_S, requested) if requested > 0 else 0.0


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


def waypoint_timeout_info(position, waypoint, speed_scale, perception_config):
    return timeout_info(position, waypoint, speed_scale, perception_config,
                        MAX_HORIZONTAL_SPEED_M_S, MIN_RISK_SPEED_M_S, WAYPOINT_TIMEOUT_MODE)


def print_waypoint_timeout_debug(
    waypoint, latest, perception_config, perception_detector, command
):
    position = local_position(latest)
    detection = current_perception_detection(
        perception_config, perception_detector, position, latest["attitude"]
    )
    print_timeout_debug(waypoint, position, target_errors(position, waypoint),
                        detection, horizontal_command_speed(command))


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
    loop = asyncio.get_running_loop()
    watchdog = WaypointProgressWatchdog(
        loop.time(), timeout_info["timeout_s"], timeout_info["distance_m"],
        enabled=WAYPOINT_TIMEOUT_MODE == "auto",
    )
    while watchdog.should_continue(loop.time()):
        ensure_critical_telemetry_fresh(latest, TELEMETRY_TIMEOUT_S)
        position = local_position(latest)
        if position is None:
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            await asyncio.sleep(0.2)
            continue
        error = target_errors(position, waypoint)
        watchdog.observe(error["horizontal_m"], loop.time())
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
        replacement_waypoints = await active_replan_replacement(
            drone, phase_state, replan_config, replan_state, position,
            detection, risk_level, route_direction, now_s,
        )
        if replacement_waypoints:
            return replacement_waypoints
        if (
            safety_decision
            and safety_decision.action == "replan_or_hover"
            and not following_escape_route
            and (
                route_direction != "return"
                or return_has_geometric_obstacle_evidence(route_direction, detection)
            )
        ):
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            await asyncio.sleep(0.2)
            continue
        semantic_safety_active = bool(
            safety_decision
            and safety_decision.action in {
                "hover", "hover_then_land", "replan_or_hover",
            }
        )
        replan_state["safety_replan_active"] = semantic_safety_active
        semantic_replacement = await active_semantic_replacement(
            drone,
            phase_state,
            replan_config,
            replan_state,
            safety_replan_active=semantic_safety_active,
        )
        if semantic_replacement:
            return semantic_replacement
        risk_action = perception_config.get("risk_action", "log_only")
        if risk_action == "stop_and_land" and risk_level == "danger":
            last_command = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(last_command)
            raise DangerObstacleDetected
        if (
            error["horizontal_m"] < REACHED_HORIZONTAL_ERROR_M
            and abs(error["down_m"]) < REACHED_VERTICAL_ERROR_M
        ):
            last_command = VelocityNedYaw(
                0.0,
                0.0,
                0.0,
                normalize_yaw_deg(waypoint.get("yaw_deg", 0.0)),
            )
            await drone.offboard.set_velocity_ned(last_command)
            print(f"Reached {waypoint['name']}.")
            follow_up = await complete_semantic_waypoint(drone, phase_state, replan_config, replan_state, waypoint, asyncio.get_running_loop().time())
            dwell_s = waypoint_dwell_s(waypoint)
            if dwell_s > 0:
                await asyncio.sleep(dwell_s)
            return follow_up
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
    print(f"  timeout reason: {watchdog.stop_reason(loop.time())}")
    return finish_or_raise_waypoint_timeout(
        waypoint, target_errors(local_position(latest), waypoint),
        REACHED_HORIZONTAL_ERROR_M, REACHED_VERTICAL_ERROR_M,
    )


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
        if replacement_waypoints and (replan_config.get("mode") == "active" or replan_config.get("semantic_runtime_mode") == "active_semantic_inspection"):
            active_waypoints = list(replacement_waypoints)
            waypoint_index = 0
            publish_dynamic_replan_event(
                phase_state,
                replan_config,
                "dynamic_replan_resumed",
                waypoint_count=len(active_waypoints),
            )
            print(
                f"Continuing {route_direction} flight on active replanned route "
                f"with {len(active_waypoints)} waypoint(s)."
            )
            continue
        completed_waypoints.append(active_waypoints[waypoint_index])
        waypoint_index += 1
    if (
        replan_config.get("semantic_runtime_mode") == "active_semantic_inspection"
        and not replan_state.get("semantic_mission_complete")
    ):
        replan_config.setdefault("semantic_feedback", []).append({
            "event": "waypoint_reached",
            "trigger": "waypoint_reached",
            "route_kind": replan_state.get("semantic_active_kind", "bootstrap"),
            "timestamp_s": asyncio.get_running_loop().time(),
        })
        publish_mission_event(
            phase_state,
            "planner_feedback_received",
            feedback="waypoint_reached",
            route_kind=replan_state.get("semantic_active_kind", "bootstrap"),
        )
        while not replan_state.get("semantic_mission_complete"):
            replacement = await active_semantic_replacement(
                drone, phase_state, replan_config, replan_state,
                safety_replan_active=bool(replan_state.get("safety_replan_active")),
            )
            if replacement:
                return completed_waypoints + await fly_waypoint_route(
                    drone, latest, phase_state, target_state, replacement,
                    phase_name, route_direction, perception_config,
                    perception_detector, replan_config, replan_state, speed_scale,
                )
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(0.2)
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
    route_altitude_m = abs(float(waypoints[0]["down_m"]))
    target_takeoff_altitude_m = max(2.5, route_altitude_m)
    print(f"Setting PX4 takeoff altitude to {target_takeoff_altitude_m:.2f} m...")
    await drone.action.set_takeoff_altitude(target_takeoff_altitude_m)
    set_phase(phase_state, "takeoff")
    print("Arming...")
    await arm_when_ready(drone.action)
    print("Taking off...")
    await drone.action.takeoff()
    print("Waiting for a continuous stable takeoff hover...")
    await wait_for_local_position(latest, TELEMETRY_TIMEOUT_S)
    await wait_for_takeoff_hover(latest, target_takeoff_altitude_m, TELEMETRY_TIMEOUT_S)
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
