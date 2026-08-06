"""Bounded MAVSDK startup with a fresh server for each attempt."""

import asyncio


DEFAULT_STARTUP_ATTEMPTS = 2
DEFAULT_RETRY_DELAY_S = 2.0


async def connect_mavsdk(
    system_factory,
    system_address,
    timeout_s,
    wait_for_connection,
    close_system,
    *,
    attempts=DEFAULT_STARTUP_ATTEMPTS,
    retry_delay_s=DEFAULT_RETRY_DELAY_S,
):
    """Return a connected system, retrying transient startup failures once."""
    if attempts < 1:
        raise ValueError("MAVSDK startup attempts must be at least one")
    last_error = None
    for attempt in range(1, attempts + 1):
        drone = system_factory()
        try:
            print(
                f"MAVSDK connection attempt {attempt}/{attempts} "
                f"at {system_address}..."
            )
            await asyncio.wait_for(
                drone.connect(system_address=system_address), timeout=timeout_s
            )
            await wait_for_connection(drone, timeout_s)
            return drone
        except (TimeoutError, ConnectionError, OSError) as error:
            if isinstance(error, TimeoutError) and not str(error):
                error = TimeoutError(
                    f"Timed out waiting {timeout_s:g}s for MAVSDK server startup"
                )
            last_error = error
            close_system(drone)
            if attempt == attempts:
                break
            print(
                f"MAVSDK attempt {attempt}/{attempts} failed: {error}. "
                f"Retrying with a fresh server in {retry_delay_s:g}s..."
            )
            await asyncio.sleep(retry_delay_s)
        except Exception:
            close_system(drone)
            raise
    raise TimeoutError(
        f"MAVSDK connection failed after {attempts} attempts; "
        f"last error: {last_error}"
    ) from last_error
