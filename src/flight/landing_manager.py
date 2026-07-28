"""Normal and best-effort failsafe landing operations."""

import asyncio
import contextlib

from src.flight.flight_state import set_phase


LANDING_TIMEOUT_S = 45.0


def configure_landing_timeout(timeout_s):
    global LANDING_TIMEOUT_S
    LANDING_TIMEOUT_S = timeout_s


async def wait_until_landed(latest, timeout_s=45):
    print("Waiting for landing to finish...")
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if latest["in_air"] is False:
            print("Drone has landed.")
            return True
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
