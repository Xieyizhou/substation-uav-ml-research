import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from src.flight.startup_gate import startup_state,wait_px4_startup

class StartupGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_explicit_success_allows_connection(self):
        self.assertFalse(startup_state('commander started'))
        self.assertTrue(startup_state('INFO [px4] Startup script returned successfully'))
        with self.assertRaises(RuntimeError):startup_state('Startup script returned with return value: 2')
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'px4.log';log.write_text('Startup script returned successfully')
            self.assertLess(await wait_px4_startup(log,[SimpleNamespace(returncode=None)]),1)
            with self.assertRaises(RuntimeError):await wait_px4_startup(log,[SimpleNamespace(returncode=1)])
            with self.assertRaises(ValueError):await wait_px4_startup(log,[],timeout_s=91)
            def cancel():raise RuntimeError('stop requested')
            with self.assertRaisesRegex(RuntimeError,'stop requested'):await wait_px4_startup(log,[],check_cancel=cancel)

    def test_reused_flights_have_identical_flight_controller(self):
        def follow(path):
            tree=ast.parse(Path(path).read_text())
            return ast.dump(next(node for node in ast.walk(tree) if isinstance(node,ast.AsyncFunctionDef) and node.name=='follow'))
        self.assertEqual(follow('scripts/flight/fly_lowload_replan.py'),follow('scripts/flight/fly_startup_replan.py'))
