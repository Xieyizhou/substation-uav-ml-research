import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.flight import telemetry_runtime


async def one_value(value):
    yield value


class FakeCore:
    def connection_state(self):
        return one_value(SimpleNamespace(is_connected=True))


class FakeTelemetry:
    def position_velocity_ned(self):
        position = SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.5)
        velocity = SimpleNamespace(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0)
        return one_value(SimpleNamespace(position=position, velocity=velocity))

    def attitude_euler(self):
        return one_value(SimpleNamespace(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0))

    def battery(self):
        return one_value(SimpleNamespace(remaining_percent=0.8))

    def flight_mode(self):
        return one_value("OFFBOARD")

    def armed(self):
        return one_value(True)

    def in_air(self):
        return one_value(True)


class FakeDrone:
    core = FakeCore()
    telemetry = FakeTelemetry()


class TelemetryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_telemetry_writes_header_and_row_and_resets_event(self):
        stop = __import__("asyncio").Event()
        latest = {
            "connected": True,
            "position_velocity": None,
            "attitude": None,
            "battery": None,
            "flight_mode": "",
            "armed": False,
            "updated_at": {},
        }
        replan_state = {"replan_triggered": True, "replan_route_replaced": False}

        async def stop_after_cycle(_seconds):
            stop.set()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            with (
                patch.object(telemetry_runtime, "LOG_DIR", Path(directory)),
                patch.object(telemetry_runtime.asyncio, "sleep", new=stop_after_cycle),
                patch.object(
                    telemetry_runtime,
                    "current_perception_detection",
                    return_value=None,
                ),
            ):
                await telemetry_runtime.log_telemetry(
                    FakeDrone(),
                    stop,
                    path,
                    latest,
                    {"phase": "flight", "route_direction": "outbound"},
                    {"name": "", "north_m": 0, "east_m": 0, "down_m": 0},
                    {
                        "grid_start": "(0,0)",
                        "grid_goal": "(2,2)",
                        "grid_width": 3,
                        "grid_height": 3,
                        "planner_name": "astar",
                        "map_name": "complex",
                        "resolution_m": 1.0,
                        "altitude_m": 1.5,
                        "return_home_enabled": False,
                    },
                    {"enabled": False},
                    {"enabled": False},
                    replan_state,
                )
            with path.open() as source:
                rows = list(csv.reader(source))

        self.assertEqual(len(rows), 2)
        self.assertIn("elapsed_s", rows[0])
        self.assertFalse(replan_state["replan_triggered"])


if __name__ == "__main__":
    unittest.main()
