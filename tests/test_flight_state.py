import asyncio
from types import SimpleNamespace
import unittest

from src.flight.flight_state import ensure_critical_telemetry_fresh


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


if __name__ == "__main__":
    unittest.main()
