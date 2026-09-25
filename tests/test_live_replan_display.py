import json
from pathlib import Path
import tempfile
import time
import unittest
from src.sandbox.live_replan_display import flight_display
from src.sandbox.live_replan_gate import BASE

class DisplayTests(unittest.TestCase):
    def test_missing_and_partial_tail_never_fabricate_telemetry(self):
        with tempfile.TemporaryDirectory() as d:
            run='sandbox-replan-v1-'+'a'*32;job=dict(scenario_id=run,job_id='test',state='running')
            self.assertIsNone(flight_display(d,job)['local_position'])
            path=Path(d)/BASE/run/'runtime/telemetry.jsonl';path.parent.mkdir(parents=True)
            row=dict(stream='local',monotonic=time.monotonic(),phase='post_hover_trial',value=dict(north=1,east=2,down=-2,vn=.1,ve=0))
            path.write_text(json.dumps(row)+'\n{"partial":')
            result=flight_display(d,job)
            self.assertEqual(result['phase'],'post_hover_trial');self.assertEqual(result['speed_m_s'],.1)
            self.assertTrue(result['display_only'])
