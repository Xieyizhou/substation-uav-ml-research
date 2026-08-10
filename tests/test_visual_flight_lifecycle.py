import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from src.flight.flight_state import set_phase
from src.flight.mission_events import MissionEventWriter
from src.vision.collection.flight_lifecycle import (
    monitor_flight_lifecycle,
    visual_phase_for_flight_event,
    write_live_status,
)


class FlightEventWriterTests(unittest.TestCase):
    def test_phase_changes_are_published_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "flight.jsonl")
            writer = MissionEventWriter(path)
            state = {
                "phase": "connecting",
                "route_direction": "none",
                "_event_publisher": writer,
            }
            set_phase(state, "outbound_to_goal", "outbound")
            set_phase(state, "outbound_to_goal", "outbound")
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_type"], "phase_changed")
            self.assertEqual(rows[0]["event_sequence"], 1)

    def test_visual_phase_mapping_uses_final_outbound_waypoint(self):
        self.assertEqual(
            visual_phase_for_flight_event(
                {
                    "event_type": "waypoint_started",
                    "route_direction": "outbound",
                    "is_final_waypoint": True,
                }
            ),
            "approach",
        )
        self.assertIsNone(
            visual_phase_for_flight_event(
                {
                    "event_type": "waypoint_started",
                    "route_direction": "return",
                    "is_final_waypoint": True,
                }
            )
        )


class FlightLifecycleMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for_rows(self, path, count):
        for _ in range(100):
            if path.is_file() and len(path.read_text().splitlines()) >= count:
                return
            await asyncio.sleep(0.005)
        self.fail(f"timed out waiting for {count} visual events")

    async def test_four_phases_and_confirmed_landing_stop_automatically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flight_path = root / "flight.jsonl"
            visual_path = root / "visual.jsonl"
            live_path = root / "live.json"
            writer = MissionEventWriter(flight_path)
            stop = asyncio.Event()
            outcome = {}
            monitor = asyncio.create_task(
                monitor_flight_lifecycle(
                    flight_path,
                    live_path,
                    visual_path,
                    stop,
                    outcome,
                    post_landing_drain_s=0,
                    poll_interval_s=0.005,
                )
            )
            events = (
                (
                    1.0,
                    "phase_changed",
                    {
                        "phase": "outbound_to_goal",
                        "route_direction": "outbound",
                    },
                ),
                (
                    3.0,
                    "waypoint_started",
                    {
                        "phase": "outbound_to_goal",
                        "route_direction": "outbound",
                        "is_final_waypoint": True,
                    },
                ),
                (5.0, "phase_changed", {"phase": "goal_hover"}),
                (8.0, "phase_changed", {"phase": "return_to_start"}),
            )
            for index, (timestamp, event_type, details) in enumerate(events, 1):
                write_live_status(
                    live_path,
                    {"last_simulation_timestamp": timestamp},
                )
                writer.publish(event_type, **details)
                await self._wait_for_rows(visual_path, index)
            write_live_status(
                live_path,
                {"last_simulation_timestamp": 12.0},
            )
            writer.publish(
                "mission_completed",
                status="completed",
                phase="landed",
                landing_confirmed=True,
            )
            await asyncio.wait_for(monitor, timeout=1)
            rows = [
                json.loads(line)
                for line in visual_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["mission_phase"] for row in rows],
                [
                    "cruise_distant",
                    "approach",
                    "close_inspection",
                    "target_transition",
                ],
            )
            self.assertEqual(
                [row["simulation_timestamp"] for row in rows],
                [1.0, 3.0, 5.0, 8.0],
            )
            self.assertTrue(stop.is_set())
            self.assertEqual(outcome["event_type"], "mission_completed")

    async def test_failed_flight_stops_without_completed_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flight_path = root / "flight.jsonl"
            writer = MissionEventWriter(flight_path)
            stop = asyncio.Event()
            outcome = {}
            monitor = asyncio.create_task(
                monitor_flight_lifecycle(
                    flight_path,
                    root / "live.json",
                    root / "visual.jsonl",
                    stop,
                    outcome,
                    post_landing_drain_s=0,
                    poll_interval_s=0.005,
                )
            )
            writer.publish(
                "mission_failed",
                status="failed",
                phase="landing_after_error",
                landing_confirmed=True,
                message="synthetic failure",
            )
            await asyncio.wait_for(monitor, timeout=1)
            self.assertTrue(stop.is_set())
            self.assertEqual(outcome["event_type"], "mission_failed")

    async def test_event_waits_until_rgb_receive_clock_reaches_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flight_path = root / "flight.jsonl"
            visual_path = root / "visual.jsonl"
            live_path = root / "live.json"
            writer = MissionEventWriter(flight_path)
            stop = asyncio.Event()
            monitor = asyncio.create_task(
                monitor_flight_lifecycle(
                    flight_path,
                    live_path,
                    visual_path,
                    stop,
                    {},
                    poll_interval_s=0.005,
                )
            )
            writer.publish("phase_changed", phase="approach")
            event = json.loads(flight_path.read_text().splitlines()[0])
            host_s = event["host_monotonic_ns"] / 1_000_000_000.0
            write_live_status(
                live_path,
                {
                    "last_simulation_timestamp": 10.0,
                    "last_receive_monotonic_timestamp": host_s - 1.0,
                },
            )
            await asyncio.sleep(0.03)
            self.assertFalse(visual_path.exists())
            write_live_status(
                live_path,
                {
                    "last_simulation_timestamp": 12.0,
                    "last_receive_monotonic_timestamp": host_s + 0.01,
                },
            )
            await self._wait_for_rows(visual_path, 1)
            row = json.loads(visual_path.read_text().splitlines()[0])
            self.assertEqual(row["simulation_timestamp"], 12.0)
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
