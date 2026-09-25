from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch
import threading
from src.sandbox.live_replan_gate import BASE
from src.sandbox.live_replan_stop import ACTION,cooperative_stop,check_stop,validate_run_id

class ControlledStopTests(unittest.TestCase):
    def test_stop_request_is_scoped_and_observable(self):
        with tempfile.TemporaryDirectory() as d:
            config=SimpleNamespace(project_root=Path(d))
            run='sandbox-replan-v1-'+'a'*32
            job=SimpleNamespace(action=ACTION,scenario_id=run,job_id='test-job',error=None)
            self.assertTrue(cooperative_stop(config,job,lambda:True))
            self.assertTrue((Path(d)/BASE/run/'stop-requested.json').exists())
            with self.assertRaisesRegex(RuntimeError,'controlled stop'):check_stop(Path(d)/BASE/run)
            self.assertFalse(cooperative_stop(config,job,lambda:False,timeout_s=0))

    def test_unrelated_jobs_and_unsafe_identifiers(self):
        self.assertFalse(cooperative_stop(None,SimpleNamespace(action='doctor'),lambda:False))
        for value in (None,'../other','sandbox-replan-v1-../other','sandbox-replan-v1-'+'a'*31):
            with self.assertRaises(ValueError):validate_run_id(value)

    def test_cooperative_completion_does_not_kill_simulator(self):
        from src.sandbox.job_runtime import stop_managed_process
        process=Mock();job=SimpleNamespace(action=ACTION)
        with patch('src.sandbox.job_runtime.cooperative_stop',return_value=True),patch('src.sandbox.job_runtime.stop_job_process') as force:
            stop_managed_process(process,job,None);force.assert_not_called()
        with patch('src.sandbox.job_runtime.cooperative_stop',return_value=False),patch('src.sandbox.job_runtime.stop_job_process') as force:
            stop_managed_process(process,job,None);force.assert_called_once_with(process)

    def test_app_shutdown_allows_bounded_landing_cleanup(self):
        from src.sandbox.operator import SandboxOperator
        operator=SandboxOperator.__new__(SandboxOperator)
        operator._guard=threading.RLock();operator._active=SimpleNamespace(action=ACTION,job_id='owned')
        operator._thread=Mock();operator.stop=Mock()
        operator.shutdown();operator.stop.assert_called_once_with('owned');operator._thread.join.assert_called_once_with(timeout=130.)
