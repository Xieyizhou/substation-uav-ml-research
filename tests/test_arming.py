import unittest
from unittest.mock import AsyncMock, patch

from src.flight.arming import arm_when_ready


class ArmingTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_command_denial(self):
        action = AsyncMock()
        action.arm.side_effect = [RuntimeError("COMMAND_DENIED"), None]
        with patch("src.flight.arming.asyncio.sleep", new=AsyncMock()) as sleep:
            await arm_when_ready(action, retry_delay_s=0.0)
        self.assertEqual(action.arm.await_count, 2)
        sleep.assert_awaited_once_with(0.0)

    async def test_preserves_non_prearm_failure(self):
        action = AsyncMock()
        action.arm.side_effect = RuntimeError("connection lost")
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            await arm_when_ready(action)
        action.arm.assert_awaited_once()

    async def test_stops_after_bounded_denials(self):
        action = AsyncMock()
        action.arm.side_effect = RuntimeError("Command Denied")
        with patch("src.flight.arming.asyncio.sleep", new=AsyncMock()):
            with self.assertRaisesRegex(RuntimeError, "Command Denied"):
                await arm_when_ready(action, attempts=2, retry_delay_s=0.0)
        self.assertEqual(action.arm.await_count, 2)


if __name__ == "__main__":
    unittest.main()
