"""MAVSDK connection and position-readiness waits."""

import asyncio

from src.flight.flight_state import local_position


async def wait_for_connection(drone, timeout_s):
    print("Waiting for drone connection...")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    connection_stream = drone.core.connection_state().__aiter__()
    while True:
        remaining_s = deadline - loop.time()
        if remaining_s <= 0:
            raise TimeoutError(f"Timed out waiting {timeout_s:g}s for PX4 connection")
        try:
            state = await asyncio.wait_for(connection_stream.__anext__(), timeout=remaining_s)
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                f"Timed out waiting {timeout_s:g}s for PX4 connection"
            ) from error
        except StopAsyncIteration as error:
            raise ConnectionError("PX4 connection stream ended before connecting") from error
        if state.is_connected:
            print("Connected to drone.")
            return


def health_status_text(health):
    if health is None:
        return (
            "local_position_ok=unknown, global_position_ok=unknown, "
            "home_position_ok=unknown"
        )
    return (
        f"local_position_ok={health.is_local_position_ok}, "
        f"global_position_ok={health.is_global_position_ok}, "
        f"home_position_ok={health.is_home_position_ok}"
    )


async def wait_for_position_ready(drone, timeout_s):
    print("Waiting for PX4 position readiness...")
    print("This A* experiment flies local NED waypoints, so local position is the main requirement.")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    next_status_print = 0.0
    latest_health = None
    health_stream = drone.telemetry.health().__aiter__()
    while True:
        remaining_s = deadline - loop.time()
        if remaining_s <= 0:
            raise TimeoutError(
                "Timed out waiting for local position readiness. "
                f"Latest health: {health_status_text(latest_health)}"
            )
        try:
            latest_health = await asyncio.wait_for(
                health_stream.__anext__(), timeout=remaining_s
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                "Timed out waiting for local position readiness. "
                f"Latest health: {health_status_text(latest_health)}"
            ) from error
        except StopAsyncIteration as error:
            raise ConnectionError(
                "PX4 health stream ended before local position became ready"
            ) from error
        now = loop.time()
        if now >= next_status_print:
            print(f"Position health: {health_status_text(latest_health)}")
            next_status_print = now + 3
        if latest_health.is_local_position_ok:
            if latest_health.is_global_position_ok and latest_health.is_home_position_ok:
                print("Full global/home position is ready.")
            else:
                print("Local position is OK. Continuing because this experiment uses local NED waypoints.")
            return


async def wait_for_local_position(latest, timeout_s):
    print("Waiting for local NED position telemetry...")
    deadline = asyncio.get_running_loop().time() + timeout_s
    while local_position(latest) is None:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting {timeout_s:g}s for local NED position telemetry"
            )
        await asyncio.sleep(0.2)
    print("Local NED position telemetry is available.")
