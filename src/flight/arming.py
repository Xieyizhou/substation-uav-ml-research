"""Bounded handling for transient PX4 pre-arm readiness delays."""

from __future__ import annotations

import asyncio


DEFAULT_ARM_ATTEMPTS = 3
DEFAULT_ARM_RETRY_DELAY_S = 3.0


def _is_command_denied(error):
    return "COMMAND_DENIED" in str(error) or "Command Denied" in str(error)


async def arm_when_ready(
    action, *, attempts=DEFAULT_ARM_ATTEMPTS,
    retry_delay_s=DEFAULT_ARM_RETRY_DELAY_S,
):
    """Retry only PX4's transient pre-arm denial; preserve other failures."""
    if attempts < 1:
        raise ValueError("arm attempts must be at least one")
    for attempt in range(1, attempts + 1):
        try:
            await action.arm()
            return
        except Exception as error:
            if not _is_command_denied(error) or attempt == attempts:
                raise
            print(
                f"PX4 denied arming attempt {attempt}/{attempts}; "
                f"waiting {retry_delay_s:g}s for pre-arm readiness..."
            )
            await asyncio.sleep(retry_delay_s)
