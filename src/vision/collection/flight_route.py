"""PX4 mission runner for validated visual observation routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mavsdk.offboard import VelocityNedYaw

from src.flight.flight_state import set_phase
from src.flight.landing_manager import wait_until_landed
from src.flight.mavsdk_preflight import wait_for_local_position
from src.flight.waypoint_executor import (
    fly_to_waypoint,
    fly_waypoint_route,
    hover_at_waypoint,
    takeoff_climb_waypoint,
    validate_takeoff_stability,
)
from src.flight import waypoint_executor
from src.vision.collection.route import VisualRoute


def load_visual_route(path):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing visual route: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed visual route: {path}: {error}") from error
    return VisualRoute.from_record(record)


def _cell_waypoint(cell, altitude_m, yaw_deg, name):
    return {
        "name": name,
        "north_m": float(cell[1]) + 0.5,
        "east_m": float(cell[0]) + 0.5,
        "down_m": -float(altitude_m),
        "yaw_deg": float(yaw_deg),
    }


def _observation_waypoint(observation):
    return {
        "name": observation.waypoint_id,
        "north_m": observation.north_m,
        "east_m": observation.east_m,
        "down_m": -observation.altitude_m,
        "yaw_deg": observation.yaw_deg,
    }


async def _takeoff(drone, latest, phase_state, target_state, route, configs):
    altitude_m = route.waypoints[0].altitude_m
    await drone.action.set_takeoff_altitude(altitude_m)
    set_phase(phase_state, "takeoff")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(8)
    await wait_for_local_position(latest, waypoint_executor.TELEMETRY_TIMEOUT_S)
    validate_takeoff_stability(latest, altitude_m)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()
    await fly_to_waypoint(
        drone,
        latest,
        phase_state,
        target_state,
        takeoff_climb_waypoint(latest, -altitude_m),
        "takeoff",
        "none",
        1.0,
        *configs,
    )


async def _fly_observations(drone, latest, phase_state, target_state, route, configs):
    traversed = []
    for observation in route.waypoints:
        transit = [
            _cell_waypoint(cell, observation.altitude_m, observation.yaw_deg, f"{observation.waypoint_id}_path_{index:03d}")
            for index, cell in enumerate(observation.transit_cells, start=1)
        ]
        transit.append(_observation_waypoint(observation))
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
        await hover_at_waypoint(
            drone,
            phase_state,
            target_state,
            transit[-1],
            observation.mission_phase,
            observation.hold_s,
        )
        traversed.extend(transit[:-1])
    return traversed


async def _return_and_land(drone, latest, phase_state, target_state, route, traversed, configs):
    return_route = [dict(waypoint, name=f"return_{index:03d}") for index, waypoint in enumerate(reversed(traversed), start=1)]
    return_route.append(
        _cell_waypoint(
            route.start_cell,
            route.waypoints[0].altitude_m,
            route.waypoints[-1].yaw_deg,
            "return_start",
        )
    )
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
    traversed = await _fly_observations(drone, latest, phase_state, target_state, route, configs)
    await _return_and_land(drone, latest, phase_state, target_state, route, traversed, configs)
