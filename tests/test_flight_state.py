import asyncio
from types import SimpleNamespace
import unittest

from src.flight.flight_state import (
    ensure_critical_telemetry_fresh,
    finish_or_raise_waypoint_timeout,
)


def _latest(*, connected, age_s, with_position=True):
    now = asyncio.get_running_loop().time()
    return {
        "connected": connected,
        "position_velocity": (
            SimpleNamespace(position=SimpleNamespace()) if with_position else None
        ),
        "updated_at": (
            {"position_velocity": now - age_s} if with_position else {}
        ),
    }


class CriticalTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_position_tolerates_transient_connection_state(self):
        ensure_critical_telemetry_fresh(
            _latest(connected=False, age_s=0.1), timeout_s=2.0
        )

    async def test_disconnection_fails_after_position_becomes_stale(self):
        with self.assertRaisesRegex(ConnectionError, "position telemetry became stale"):
            ensure_critical_telemetry_fresh(
                _latest(connected=False, age_s=3.0), timeout_s=2.0
            )

    async def test_connected_but_stale_position_remains_a_timeout(self):
        with self.assertRaisesRegex(TimeoutError, "Local position telemetry is stale"):
            ensure_critical_telemetry_fresh(
                _latest(connected=True, age_s=3.0), timeout_s=2.0
            )

    async def test_missing_position_remains_a_timeout(self):
        with self.assertRaisesRegex(TimeoutError, "has not been received"):
            ensure_critical_telemetry_fresh(
                _latest(connected=False, age_s=0.0, with_position=False),
                timeout_s=2.0,
            )


class WaypointTimeoutAcceptanceTests(unittest.TestCase):
    def test_accepts_only_the_bounded_horizontal_timeout_margin(self):
        waypoint = {"name": "RWP02"}
        accepted = {"horizontal_m": 0.54, "down_m": 0.01}
        self.assertIsNone(
            finish_or_raise_waypoint_timeout(waypoint, accepted, 0.5, 0.1)
        )
        with self.assertRaisesRegex(TimeoutError, "RWP02"):
            finish_or_raise_waypoint_timeout(
                waypoint, {"horizontal_m": 0.61, "down_m": 0.01}, 0.5, 0.1
            )

    def test_rejects_vertical_error_even_inside_horizontal_margin(self):
        with self.assertRaises(TimeoutError):
            finish_or_raise_waypoint_timeout(
                {"name": "WP"}, {"horizontal_m": 0.54, "down_m": 0.2},
                0.5, 0.1,
            )


if __name__ == "__main__":
    unittest.main()
