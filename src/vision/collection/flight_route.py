"""PX4 mission runner for validated visual observation routes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

from mavsdk.offboard import VelocityNedYaw

from src.flight.flight_state import publish_mission_event, set_phase
from src.flight.landing_manager import wait_until_landed
from src.flight.mavsdk_preflight import wait_for_local_position
from src.flight.takeoff_stability import (
    wait_for_ground_stability,
    wait_for_takeoff_hover,
)
from src.flight.waypoint_executor import (
    fly_to_waypoint,
    fly_waypoint_route,
    hover_at_waypoint,
    normalize_yaw_deg,
    takeoff_climb_waypoint,
    validate_takeoff_stability,
)
from src.flight import waypoint_executor
from src.vision.collection.route import VisualRoute


ACTION_TAKEOFF_ALTITUDE_M = 2.5
ACTION_TAKEOFF_SPEED_M_S = 0.5
YAW_SLEW_RATE_DEG_S = 20.0
YAW_COMMAND_INTERVAL_S = 0.2


def load_visual_route(path):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing visual route: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed visual route: {path}: {error}") from error
    return VisualRoute.from_record(record)


def _cell_waypoint(cell, altitude_m, yaw_deg, name, origin):
    origin_east_m, origin_north_m = origin
    return {
        "name": name,
        "north_m": float(cell[1]) + 0.5 - origin_north_m,
        "east_m": float(cell[0]) + 0.5 - origin_east_m,
        "down_m": -float(altitude_m),
        "yaw_deg": float(yaw_deg),
    }


def _observation_waypoint(observation, origin):
    origin_east_m, origin_north_m = origin
    return {
        "name": observation.waypoint_id,
        "north_m": observation.north_m - origin_north_m,
        "east_m": observation.east_m - origin_east_m,
        "down_m": -observation.altitude_m,
        "yaw_deg": observation.yaw_deg,
    }


def _yaw_error_deg(current, target):
    return (float(target) - float(current) + 180.0) % 360.0 - 180.0


async def _settle_observation_yaw(drone, latest, phase_state, observation, route):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + route.yaw_acquisition_timeout_s
    stable_since = None
    attitude = latest.get("attitude")
    command_yaw = (
        observation.yaw_deg if attitude is None else float(attitude.yaw_deg)
    )
    while loop.time() < deadline:
        attitude = latest.get("attitude")
        yaw_error = _yaw_error_deg(command_yaw, observation.yaw_deg)
        max_step = YAW_SLEW_RATE_DEG_S * YAW_COMMAND_INTERVAL_S
        command_yaw += max(-max_step, min(max_step, yaw_error))
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, normalize_yaw_deg(command_yaw))
        )
        settled = (
            attitude is not None
            and abs(_yaw_error_deg(attitude.yaw_deg, observation.yaw_deg))
            <= route.yaw_tolerance_deg
            and abs(float(attitude.roll_deg)) <= route.level_tolerance_deg
            and abs(float(attitude.pitch_deg)) <= route.level_tolerance_deg
        )
        if settled:
            stable_since = stable_since or loop.time()
            if loop.time() - stable_since >= route.yaw_settle_duration_s:
                publish_mission_event(
                    phase_state,
                    "yaw_settled",
                    phase=observation.mission_phase,
                    waypoint_name=observation.waypoint_id,
                    target_yaw_deg=observation.yaw_deg,
                    observed_yaw_deg=float(attitude.yaw_deg),
                    observed_roll_deg=float(attitude.roll_deg),
                    observed_pitch_deg=float(attitude.pitch_deg),
                )
                return
        else:
            stable_since = None
        await asyncio.sleep(YAW_COMMAND_INTERVAL_S)
    raise TimeoutError(
        f"Timed out waiting for yaw at {observation.waypoint_id}"
    )


async def _takeoff(drone, latest, phase_state, target_state, route, configs):
    route_altitude_m = route.waypoints[0].altitude_m
    action_altitude_m = max(ACTION_TAKEOFF_ALTITUDE_M, route_altitude_m)
    await wait_for_local_position(latest, waypoint_executor.TELEMETRY_TIMEOUT_S)
    attitude = latest.get("attitude")
    if attitude is None:
        raise RuntimeError("Visual takeoff requires attitude telemetry")
    validate_takeoff_stability(latest, action_altitude_m)
    await wait_for_ground_stability(latest, waypoint_executor.TELEMETRY_TIMEOUT_S)
    set_phase(phase_state, "takeoff")
    await drone.param.set_param_float("MPC_TKO_SPEED", ACTION_TAKEOFF_SPEED_M_S)
    publish_mission_event(
        phase_state,
        "flight_profile_configured",
        takeoff_altitude_m=action_altitude_m,
        takeoff_speed_m_s=ACTION_TAKEOFF_SPEED_M_S,
        yaw_slew_rate_deg_s=YAW_SLEW_RATE_DEG_S,
    )
    await drone.action.set_takeoff_altitude(action_altitude_m)
    await drone.action.arm()
    await drone.action.takeoff()
    await wait_for_takeoff_hover(
        latest, action_altitude_m, waypoint_executor.TELEMETRY_TIMEOUT_S
    )
    attitude = latest.get("attitude")
    takeoff_waypoint = takeoff_climb_waypoint(latest, -route_altitude_m)
    takeoff_waypoint["yaw_deg"] = float(attitude.yaw_deg)
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(0.0, 0.0, 0.0, normalize_yaw_deg(attitude.yaw_deg))
    )
    await drone.offboard.start()
    await fly_to_waypoint(
        drone,
        latest,
        phase_state,
        target_state,
        takeoff_waypoint,
        "takeoff",
        "none",
        1.0,
        *configs,
    )
    validate_takeoff_stability(latest, route_altitude_m)
    publish_mission_event(
        phase_state,
        "takeoff_completed",
        route_altitude_m=route_altitude_m,
    )


async def _fly_observations(drone, latest, phase_state, target_state, route, configs):
    origin = (route.start_cell[0] + 0.5, route.start_cell[1] + 0.5)
    for observation in route.waypoints:
        transit = [
            _cell_waypoint(
                cell,
                observation.altitude_m,
                observation.yaw_deg,
                f"{observation.waypoint_id}_path_{index:03d}",
                origin,
            )
            for index, cell in enumerate(observation.transit_cells, start=1)
        ]
        transit.append(_observation_waypoint(observation, origin))
        await fly_waypoint_route(
            drone,
            latest,
            phase_state,
            target_state,
            transit,
            observation.mission_phase,
            "outbound",
            *configs,
        )
        await _settle_observation_yaw(
            drone,
            latest,
            phase_state,
            observation,
            route,
        )
        await hover_at_waypoint(
            drone,
            phase_state,
            target_state,
            transit[-1],
            observation.mission_phase,
            observation.hold_s,
        )


async def _settle_departure_yaw(drone, latest, phase_state, route):
    departure = replace(
        route.waypoints[0],
        waypoint_id=f"departure_{route.waypoints[0].waypoint_id}",
    )
    set_phase(phase_state, departure.mission_phase)
    await _settle_observation_yaw(
        drone,
        latest,
        phase_state,
        departure,
        route,
    )


def _return_waypoints(route):
    origin = (route.start_cell[0] + 0.5, route.start_cell[1] + 0.5)
    return [
        _cell_waypoint(
            cell,
            route.waypoints[0].altitude_m,
            route.waypoints[-1].yaw_deg,
            (
                "return_start"
                if index == len(route.return_transit_cells)
                else f"return_{index:03d}"
            ),
            origin,
        )
        for index, cell in enumerate(route.return_transit_cells, start=1)
    ]


async def _return_and_land(drone, latest, phase_state, target_state, route, configs):
    return_route = _return_waypoints(route)
    if return_route:
        await fly_waypoint_route(
            drone,
            latest,
            phase_state,
            target_state,
            return_route,
            "target_transition",
            "return",
            *configs,
        )
    set_phase(phase_state, "landing")
    await drone.offboard.stop()
    await drone.action.land()
    await wait_until_landed(latest, waypoint_executor.LANDING_TIMEOUT_S)
    set_phase(phase_state, "landed")


async def fly_visual_observation_route(
    drone,
    latest,
    phase_state,
    target_state,
    route,
    perception_config,
    perception_detector,
    replan_config,
    replan_state,
    return_home=True,
):
    if not isinstance(route, VisualRoute):
        raise TypeError("visual mission requires a VisualRoute")
    configs = (perception_config, perception_detector, replan_config, replan_state)
    await _takeoff(drone, latest, phase_state, target_state, route, configs)
    await _settle_departure_yaw(drone, latest, phase_state, route)
    await _fly_observations(drone, latest, phase_state, target_state, route, configs)
    await _return_and_land(drone, latest, phase_state, target_state, route, configs)
