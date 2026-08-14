"""Normal and best-effort failsafe landing operations."""

import asyncio
import contextlib
from math import hypot

from src.flight.flight_state import (
    local_position, local_velocity, set_phase, telemetry_age_s,
)


LANDING_TIMEOUT_S = 45.0


def configure_landing_timeout(timeout_s):
    global LANDING_TIMEOUT_S
    LANDING_TIMEOUT_S = timeout_s


def stable_ground_state(latest):
    position = local_position(latest)
    velocity = local_velocity(latest)
    flight_mode = str(latest.get("flight_mode", "")).upper()
    position_age_s = telemetry_age_s(latest, "position_velocity")
    if (
        position is None or velocity is None or position_age_s is None
        or position_age_s > 2.0 or not flight_mode.endswith("LAND")
    ):
        return False
    return (
        abs(float(position.down_m)) <= 0.25
        and hypot(float(velocity.north_m_s), float(velocity.east_m_s)) <= 0.15
        and abs(float(velocity.down_m_s)) <= 0.1
    )


async def wait_until_landed(latest, timeout_s=45, *, stable_duration_s=3.0):
    print("Waiting for landing to finish...")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    stable_since = None
    while loop.time() < deadline:
        if latest["in_air"] is False:
            print("Drone has landed.")
            return True
        if stable_ground_state(latest):
            stable_since = stable_since or loop.time()
            if loop.time() - stable_since >= stable_duration_s:
                print("Drone is continuously stable at ground level.")
                return True
        else:
            stable_since = None
        await asyncio.sleep(1)
    raise TimeoutError(
        f"Landing was not confirmed within {timeout_s:g}s; PX4 may still be landing"
    )


async def attempt_safe_landing(drone, latest, phase_state, phase_name):
    set_phase(phase_state, phase_name)
    print("Trying to stop Offboard mode and land safely...")
    with contextlib.suppress(Exception):
        await drone.offboard.stop()
    try:
        await drone.action.land()
        await wait_until_landed(latest, LANDING_TIMEOUT_S)
    except Exception as error:
        print(f"Landing was not confirmed: {error}")
        return False
    set_phase(phase_state, "landed")
    return True
