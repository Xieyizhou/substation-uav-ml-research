"""Continuous readiness and stability checks around PX4 takeoff."""

from __future__ import annotations

import asyncio
from math import sqrt

from src.flight.flight_state import (
    ensure_critical_telemetry_fresh,
    local_position,
    local_velocity,
)


def validate_takeoff_stability(latest, target_altitude_m):
    position = local_position(latest)
    attitude = latest.get("attitude")
    if position is None or attitude is None:
        raise RuntimeError("Takeoff stability check requires position and attitude")
    roll_deg = abs(float(attitude.roll_deg))
    pitch_deg = abs(float(attitude.pitch_deg))
    horizontal_drift_m = sqrt(position.north_m**2 + position.east_m**2)
    altitude_m = -float(position.down_m)
    if roll_deg > 30.0 or pitch_deg > 30.0:
        raise RuntimeError(
            "Takeoff is unstable: "
            f"roll={roll_deg:.1f} deg, pitch={pitch_deg:.1f} deg"
        )
    if horizontal_drift_m > 2.5:
        raise RuntimeError(
            f"Takeoff drifted {horizontal_drift_m:.1f} m before Offboard start"
        )
    if altitude_m < -0.5 or altitude_m > target_altitude_m + 1.5:
        raise RuntimeError(
            "Takeoff altitude is outside the stability envelope: "
            f"{altitude_m:.1f} m"
        )


def takeoff_climb_waypoint(latest, target_down_m):
    position = local_position(latest)
    if position is None:
        raise RuntimeError("Takeoff climb requires local position")
    return {
        "name": "TAKEOFF_CLIMB",
        "north_m": float(position.north_m),
        "east_m": float(position.east_m),
        "down_m": float(target_down_m),
    }


def _motion_is_stable(latest, *, max_level_deg, max_horizontal_speed, max_vertical_speed):
    attitude = latest.get("attitude")
    velocity = local_velocity(latest)
    if attitude is None or velocity is None:
        return False
    horizontal_speed = sqrt(velocity.north_m_s**2 + velocity.east_m_s**2)
    return (
        abs(float(attitude.roll_deg)) <= max_level_deg
        and abs(float(attitude.pitch_deg)) <= max_level_deg
        and horizontal_speed <= max_horizontal_speed
        and abs(float(velocity.down_m_s)) <= max_vertical_speed
    )


async def _wait_for_stable_window(
    latest, predicate, *, telemetry_timeout_s, timeout_s, stable_duration_s,
    stage,
):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    stable_since = None
    while loop.time() < deadline:
        ensure_critical_telemetry_fresh(latest, telemetry_timeout_s)
        if predicate():
            stable_since = stable_since or loop.time()
            if loop.time() - stable_since >= stable_duration_s:
                return
        else:
            stable_since = None
        await asyncio.sleep(0.2)
    raise TimeoutError(
        f"Vehicle did not reach a continuous stable {stage} window"
    )


async def wait_for_ground_stability(latest, telemetry_timeout_s, *, timeout_s=10.0):
    def ready():
        validate_takeoff_stability(latest, 0.5)
        position = local_position(latest)
        return (
            abs(float(position.down_m)) <= 0.25
            and _motion_is_stable(
                latest, max_level_deg=5.0,
                max_horizontal_speed=0.1, max_vertical_speed=0.1,
            )
        )

    await _wait_for_stable_window(
        latest, ready, telemetry_timeout_s=telemetry_timeout_s,
        timeout_s=timeout_s, stable_duration_s=2.0, stage="ground",
    )


async def wait_for_takeoff_hover(
    latest, target_altitude_m, telemetry_timeout_s, *, timeout_s=60.0
):
    minimum_altitude_m = max(0.75, target_altitude_m * 0.6)

    def ready():
        validate_takeoff_stability(latest, target_altitude_m)
        position = local_position(latest)
        return (
            -float(position.down_m) >= minimum_altitude_m
            and _motion_is_stable(
                latest, max_level_deg=5.0,
                max_horizontal_speed=0.25, max_vertical_speed=0.25,
            )
        )

    await _wait_for_stable_window(
        latest, ready, telemetry_timeout_s=telemetry_timeout_s,
        timeout_s=timeout_s, stable_duration_s=2.0, stage="takeoff hover",
    )
