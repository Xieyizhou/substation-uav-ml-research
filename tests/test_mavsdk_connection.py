import asyncio
import unittest

from src.flight.mavsdk_connection import connect_mavsdk


class FakeDrone:
    def __init__(self, *, connect_error=None):
        self.connect_error = connect_error
        self.closed = False

    async def connect(self, system_address):
        self.system_address = system_address
        if self.connect_error is not None:
            raise self.connect_error


class MavsdkConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_timeout_retries_with_fresh_system(self):
        drones = [FakeDrone(connect_error=TimeoutError("server unavailable")), FakeDrone()]

        def factory():
            return drones.pop(0)

        async def wait_for_connection(drone, timeout_s):
            self.assertEqual(timeout_s, 0.1)

        def close_system(drone):
            drone.closed = True

        first = drones[0]
        connected = await connect_mavsdk(
            factory,
            "udp://test",
            0.1,
            wait_for_connection,
            close_system,
            retry_delay_s=0,
        )
        self.assertTrue(first.closed)
        self.assertFalse(connected.closed)
        self.assertEqual(connected.system_address, "udp://test")

    async def test_connection_stream_timeout_is_retried(self):
        created = []

        def factory():
            drone = FakeDrone()
            created.append(drone)
            return drone

        calls = 0

        async def wait_for_connection(drone, timeout_s):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("PX4 connection unavailable")

        def close_system(drone):
            drone.closed = True

        connected = await connect_mavsdk(
            factory,
            "udp://test",
            0.1,
            wait_for_connection,
            close_system,
            retry_delay_s=0,
        )
        self.assertEqual(len(created), 2)
        self.assertTrue(created[0].closed)
        self.assertIs(connected, created[1])

    async def test_exhaustion_closes_every_attempt(self):
        created = []

        def factory():
            drone = FakeDrone(connect_error=TimeoutError("server unavailable"))
            created.append(drone)
            return drone

        with self.assertRaisesRegex(TimeoutError, "failed after 2 attempts"):
            await connect_mavsdk(
                factory,
                "udp://test",
                0.1,
                lambda drone, timeout: asyncio.sleep(0),
                lambda drone: setattr(drone, "closed", True),
                retry_delay_s=0,
            )
        self.assertEqual(len(created), 2)
        self.assertTrue(all(drone.closed for drone in created))


if __name__ == "__main__":
    unittest.main()
