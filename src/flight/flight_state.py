"""Small helpers for mutable mission and telemetry state."""

import asyncio
from math import sqrt


def set_phase(phase_state, phase, route_direction="none"):
    changed = (
        phase_state.get("phase") != phase
        or phase_state.get("route_direction") != route_direction
    )
    phase_state["phase"] = phase
    phase_state["route_direction"] = route_direction
    if changed:
        publish_mission_event(
            phase_state,
            "phase_changed",
            phase=phase,
            route_direction=route_direction,
        )


def publish_mission_event(phase_state, event_type, **details):
    publisher = phase_state.get("_event_publisher")
    if publisher is None:
        return None
    return publisher.publish(event_type, **details)


def update_latest(latest, key, value):
    latest[key] = value
    latest["updated_at"][key] = asyncio.get_running_loop().time()


def telemetry_age_s(latest, key):
    updated_at = latest.get("updated_at", {}).get(key)
    if updated_at is None:
        return None
    return asyncio.get_running_loop().time() - updated_at


def ensure_critical_telemetry_fresh(latest, timeout_s):
    if latest.get("connected") is False:
        raise ConnectionError("PX4 connection was lost during flight")
    age_s = telemetry_age_s(latest, "position_velocity")
    if age_s is None:
        raise TimeoutError("Local position telemetry has not been received")
    if age_s > timeout_s:
        raise TimeoutError(
            f"Local position telemetry is stale ({age_s:.1f}s > {timeout_s:.1f}s)"
        )


def value_or_blank(message, attribute):
    if message is None:
        return ""
    return getattr(message, attribute, "")


def local_position(latest):
    position_velocity = latest["position_velocity"]
    if position_velocity is None:
        return None
    return getattr(position_velocity, "position", None)


def local_velocity(latest):
    position_velocity = latest["position_velocity"]
    if position_velocity is None:
        return None
    return getattr(position_velocity, "velocity", None)


def target_errors(position, target):
    if position is None or target["name"] == "":
        return None
    error_north = target["north_m"] - position.north_m
    error_east = target["east_m"] - position.east_m
    error_down = target["down_m"] - position.down_m
    return {
        "north_m": error_north,
        "east_m": error_east,
        "down_m": error_down,
        "horizontal_m": sqrt(error_north**2 + error_east**2),
    }


def horizontal_distance_to_waypoint(position, waypoint):
    if position is None:
        return None
    north_error = waypoint["north_m"] - position.north_m
    east_error = waypoint["east_m"] - position.east_m
    return sqrt(north_error**2 + east_error**2)


def horizontal_command_speed(command):
    if command is None:
        return None
    north_m_s = getattr(command, "north_m_s", None)
    east_m_s = getattr(command, "east_m_s", None)
    if north_m_s is None or east_m_s is None:
        return None
    return sqrt(north_m_s**2 + east_m_s**2)
