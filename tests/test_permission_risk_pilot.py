import asyncio
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch,AsyncMock
from scripts.vision import resume_supervision_risk_pilot as pilot
from scripts.vision.review_permission_risk_pilot import validate_alignment

class AlignmentTests(unittest.TestCase):
    def fixture(self):
        target={'review_id':'T30','visible_pixel_count':41,'visible_bbox_xyxy':[0,428,6,435]}
        return dict(status='original_pixel_evidence_certified',process_cleanup_complete=True,stable_frames=3,
            records=[dict(rgb_exact=True,maximum_box_delta_px=0,skew_ms=32,targets=[dict(target)]) for _ in range(3)])

    def test_nonzero_small_fragment_is_not_automatically_accepted(self):
        t=validate_alignment(self.fixture(),'T30');self.assertEqual(t['visible_pixel_count'],41)
        self.assertNotIn('training_admitted',t)

    def test_inexact_unstable_empty_or_missing_evidence_rejected(self):
        cases=[]
        for field,value in [('rgb_exact',False),('skew_ms',34),('maximum_box_delta_px',1.01)]:
            r=self.fixture();r['records'][0][field]=value;cases.append(r)
        r=self.fixture();r['records'].pop();cases.append(r)
        r=self.fixture();r['records'][1]['targets'][0]['visible_pixel_count']=40;cases.append(r)
        r=self.fixture()
        for frame in r['records']:frame['targets'][0]['visible_pixel_count']=0
        cases.append(r)
        for r in cases:
            with self.assertRaises(ValueError):validate_alignment(r,'T30')

class PilotTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_technical_attempts_max(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'OUT',Path(tmp)),patch.object(pilot.replay,'attempt',new_callable=AsyncMock,return_value={'status':'technical_failure','process_cleanup_complete':True}) as call:
            r=await pilot.unit({'event_id':'T30'})
            self.assertEqual(call.call_count,3);self.assertEqual(r['status'],'technical_attempts_exhausted')

    async def test_semantic_and_alignment_results_not_retried(self):
        for status in ('semantic_blocked','alignment_held','original_pixel_evidence_certified'):
            with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'OUT',Path(tmp)),patch.object(pilot.replay,'attempt',new_callable=AsyncMock,return_value={'status':status,'process_cleanup_complete':True}) as call:
                self.assertEqual((await pilot.unit({'event_id':'T30'}))['status'],status);self.assertEqual(call.call_count,1)

    async def test_permission_preflight_does_not_consume_capture_budget(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'OUT',Path(tmp)),patch.object(pilot,'read',return_value={}),patch.object(pilot,'verify'),patch.object(pilot,'probe_loopback',side_effect=PermissionError('denied')),patch.object(pilot.replay,'attempt',new_callable=AsyncMock) as call:
            with self.assertRaises(PermissionError):await pilot.run()
            call.assert_not_called();self.assertFalse((Path(tmp)/'replay').exists())

    async def test_permission_failure_stops_after_first_attempt(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'OUT',Path(tmp)):
            folder=Path(tmp)/'replay/T30/attempt-01';folder.mkdir(parents=True)
            # Existing incomplete directories consume their slot; second attempt reports failure.
            async def failed(f,n):
                target=Path(tmp)/'replay/T30'/f'attempt-{n:02}';target.mkdir()
                (target/'simulator.log').write_text('Operation not permitted')
                return {'status':'technical_failure','process_cleanup_complete':True}
            with patch.object(pilot.replay,'attempt',side_effect=failed) as call:
                await pilot.unit({'event_id':'T30'});self.assertEqual(call.call_count,1);self.assertEqual(call.call_args.args[1],2)

    async def test_cleanup_failure_blocks(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pilot,'OUT',Path(tmp)),patch.object(pilot.replay,'attempt',new_callable=AsyncMock,return_value={'status':'technical_failure','process_cleanup_complete':False}):
            with self.assertRaises(ValueError):await pilot.unit({'event_id':'T30'})

if __name__=='__main__':unittest.main()
