import json
import tempfile
import unittest
from asyncio import sleep
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.flight import fly_astar_path
from src.flight.fly_astar_path import (
    run_flight,
    status_path_for_log,
    wait_for_connection,
    wait_for_local_position,
    wait_until_landed,
    write_run_status,
)


class NeverConnectingCore:
    async def connection_state(self):
        if False:
            yield None


class NeverConnectingDrone:
    core = NeverConnectingCore()


class RuntimeTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_wait_times_out(self):
        with self.assertRaisesRegex(TimeoutError, "PX4 connection"):
            await wait_for_connection(NeverConnectingDrone(), timeout_s=0)

    async def test_local_position_wait_times_out(self):
        latest = {"position_velocity": None}
        with self.assertRaisesRegex(TimeoutError, "local NED position"):
            await wait_for_local_position(latest, timeout_s=0)

    async def test_landing_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(TimeoutError, "Landing was not confirmed"):
            await wait_until_landed({"in_air": True}, timeout_s=0)

    async def test_landing_confirmation_succeeds(self):
        self.assertTrue(await wait_until_landed({"in_air": False}, timeout_s=1))


class RunStatusTests(unittest.TestCase):
    def test_status_is_written_atomically_with_landing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "astar_20260715_120000.csv"
            path = write_run_status(
                log_path,
                "failed",
                "landing_after_error",
                message="TimeoutError: telemetry stale",
                landing_confirmed=False,
            )
            self.assertEqual(path, status_path_for_log(log_path))
            payload = json.loads(path.read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertFalse(payload["landing_confirmed"])
            self.assertEqual(payload["run_id"], "as_20260715_120000")
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())


class FakeDrone:
    async def connect(self, system_address):
        self.system_address = system_address


class HangingConnectDrone:
    async def connect(self, system_address):
        await sleep(10)


async def logger_until_stopped(
    drone,
    stop_event,
    log_path,
    latest,
    phase_state,
    target_state,
    planner_info,
    perception_config,
    replan_config,
    replan_state,
    perception_detector,
):
    while not stop_event.is_set():
        await sleep(0)


async def successful_mission(
    drone,
    latest,
    phase_state,
    target_state,
    waypoints,
    perception_config,
    perception_detector,
    replan_config,
    replan_state,
    return_home=False,
):
    phase_state["phase"] = "landed"


async def failed_mission(*args, **kwargs):
    raise TimeoutError("telemetry stale")


class FlightOutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_mavsdk_startup_timeout_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "astar_20260715_120000.csv"
            with (
                patch.object(fly_astar_path, "System", return_value=HangingConnectDrone()),
                patch.object(fly_astar_path, "make_log_path", return_value=log_path),
                patch.object(fly_astar_path, "CONNECTION_TIMEOUT_S", 0.01),
            ):
                with self.assertRaisesRegex(TimeoutError, "MAVSDK connection startup"):
                    await run_flight(
                        "udp://test",
                        [{"name": "WP01", "north_m": 0.0, "east_m": 0.0, "down_m": -1.0}],
                        {},
                        {},
                        {},
                    )
            payload = json.loads(status_path_for_log(log_path).read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertIsNone(payload["landing_confirmed"])

    async def test_successful_mission_records_confirmed_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "astar_20260715_120001.csv"
            with (
                patch.object(fly_astar_path, "System", return_value=FakeDrone()),
                patch.object(fly_astar_path, "make_log_path", return_value=log_path),
                patch.object(fly_astar_path, "wait_for_connection", new=AsyncMock()),
                patch.object(fly_astar_path, "wait_for_position_ready", new=AsyncMock()),
                patch.object(fly_astar_path, "log_telemetry", new=logger_until_stopped),
                patch.object(fly_astar_path, "fly_astar_waypoints", new=successful_mission),
            ):
                await run_flight(
                    "udp://test",
                    [{"name": "WP01", "north_m": 0.0, "east_m": 0.0, "down_m": -1.0}],
                    {},
                    {},
                    {},
                )
            payload = json.loads(status_path_for_log(log_path).read_text())
            self.assertEqual(payload["status"], "completed")
            self.assertTrue(payload["landing_confirmed"])

    async def test_failed_mission_records_failure_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "astar_20260715_120002.csv"
            with (
                patch.object(fly_astar_path, "System", return_value=FakeDrone()),
                patch.object(fly_astar_path, "make_log_path", return_value=log_path),
                patch.object(fly_astar_path, "wait_for_connection", new=AsyncMock()),
                patch.object(fly_astar_path, "wait_for_position_ready", new=AsyncMock()),
                patch.object(fly_astar_path, "log_telemetry", new=logger_until_stopped),
                patch.object(fly_astar_path, "fly_astar_waypoints", new=failed_mission),
                patch.object(
                    fly_astar_path,
                    "attempt_safe_landing",
                    new=AsyncMock(return_value=False),
                ),
            ):
                with self.assertRaisesRegex(TimeoutError, "telemetry stale"):
                    await run_flight(
                        "udp://test",
                        [{"name": "WP01", "north_m": 0.0, "east_m": 0.0, "down_m": -1.0}],
                        {},
                        {},
                        {},
                    )
            payload = json.loads(status_path_for_log(log_path).read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertFalse(payload["landing_confirmed"])


if __name__ == "__main__":
    unittest.main()
