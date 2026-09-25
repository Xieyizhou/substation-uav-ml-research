import asyncio
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.flight.async_runtime import run_with_bounded_shutdown
from src.flight.mission_lifecycle import execute_flight


class SignalTests(unittest.TestCase):
    def test_signal_cancels_root_without_cancelling_support_task(self):
        old = signal.getsignal(signal.SIGINT)
        order = []
        async def main():
            support = asyncio.create_task(asyncio.sleep(20))
            try:
                asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGINT)
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                order.append(not support.done())
                await asyncio.sleep(0)
                order.append('recovery_finished')
            finally:
                support.cancel()
                await asyncio.gather(support, return_exceptions=True)
        run_with_bounded_shutdown(main(), .1)
        self.assertEqual(order, [True, 'recovery_finished'])
        self.assertEqual(signal.getsignal(signal.SIGINT), old)


class CancellationLandingTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, lands):
        order = []; started = asyncio.Event()
        async def mission(drone, latest, phase, *args, **kwargs):
            phase['phase'] = 'outbound'
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                order.append('route_stopped')
        async def telemetry(drone, stop, *args):
            await stop.wait()
            order.append('telemetry_stopped')
        async def land(drone, latest, phase, name):
            self.assertIn('route_stopped', order)
            self.assertNotIn('telemetry_stopped', order)
            order.append('landing_attempted')
            phase['phase'] = 'landed' if lands else name
            return lands
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp)/'events.jsonl'
            services = {'make_log_path':lambda:Path(tmp)/'telemetry.csv',
                        'write_run_status':Mock(), 'system_factory':Mock(),
                        'wait_for_connection':AsyncMock(), 'close_system':Mock(),
                        'wait_for_position_ready':AsyncMock(), 'log_telemetry':telemetry,
                        'fly_astar_waypoints':mission, 'attempt_safe_landing':land}
            settings = SimpleNamespace(connection_timeout_s=1, position_ready_timeout_s=1,
                                       logger_shutdown_timeout_s=.1)
            with patch('src.flight.mission_lifecycle.connect_mavsdk',new=AsyncMock(return_value=object())):
                task = asyncio.create_task(execute_flight('unused',[],{}, {},{},settings,services,visual_mission_events=events))
                await asyncio.wait_for(started.wait(),1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            kinds = [row['event_type'] for row in rows]
            self.assertEqual('landing_confirmed' in kinds, lands)
            self.assertIn('mission_failed', kinds)
            self.assertNotIn('mission_completed', kinds)
            self.assertEqual(order,['route_stopped','landing_attempted','telemetry_stopped'])
    async def test_cancel_lands_before_telemetry_shutdown(self):
        await self.exercise(True)
    async def test_failed_landing_never_emits_confirmation(self):
        await self.exercise(False)
