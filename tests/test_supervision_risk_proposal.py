import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,AsyncMock
from scripts.vision.supervision_risk_proposal import validate,impact,closure
from scripts.vision import supervision_risk_revision as runner
from src.ml.artifacts import object_sha256,file_sha256
from src.vision.collection.gazebo_truth import _annotation

class RiskTests(unittest.TestCase):
    def test_touching_boundary_is_not_converter_clipping(self):
        b={'label':208,'box':{'minCorner':{'x':0,'y':10},'maxCorner':{'x':15,'y':130}}}
        a=_annotation(b,index=0,width=1920,height=1080,message_id='test')
        self.assertEqual(a.truncation_status,'not_truncated');self.assertEqual(a.visibility_status,'unknown')
        b['box']['minCorner']['x']=-1
        self.assertEqual(_annotation(b,index=0,width=1920,height=1080,message_id='test').truncation_status,'truncated')

    def test_missing_duplicate_stale_or_mutating_proposal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ip=Path(tmp)/'evidence';ip.write_bytes(b'original')
            f=dict(event_id='T30',evidence_path=str(ip),evidence_sha256=file_sha256(ip))
            d=dict(event_id='T30',source_identity=object_sha256(f),evidence_sha256=f['evidence_sha256'],proposal='evidence_pending',reason='insufficient',review_nature='AI辅助审核',reviewed_at='2026-09-09',apply_label_change=False,delete_box_keep_image=False)
            p={'frames':[f]};validate(p,[d])
            for ds in ([],[d,d],[dict(d,source_identity='stale')],[dict(d,apply_label_change=True)],[dict(d,delete_box_keep_image=True)],[dict(d,proposal='label_revision_candidate')]):
                with self.assertRaises(ValueError):validate(p,ds)
            ip.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate(p,[d])

    def test_multiclass_and_zero_exposure_not_inflated(self):
        rows=[dict(member_id='a',class_instances={'reactor':1,'transformer':2}),dict(member_id='b',class_instances={'reactor':1})]
        x=impact(rows,['a','a','b'],['a']);self.assertEqual(x['image_exposures'],2)
        self.assertEqual(x['full_label_instances'],{'reactor':1,'transformer':2});self.assertEqual(x['class_instance_exposures'],{'reactor':2,'transformer':4})
        self.assertEqual(impact(rows,[],['a'])['image_exposures'],0)
        self.assertEqual(impact(rows,['a'],[])['unique_images'],0)
        with self.assertRaises(ValueError):impact(rows,['unknown'],[])

    def test_lineage_variants_whole_group_and_no_cross_map_guess(self):
        rows=[dict(member_id='a',derivation_group='pose1'),dict(member_id='b',derivation_group='pose1'),dict(member_id='c')]
        src=[dict(member_id='a',source_map='complex',source_view_id='v1'),dict(member_id='b',source_map='complex',source_view_id='v2'),dict(member_id='c',source_map='simple',source_view_id='v1')]
        self.assertEqual(closure(rows,['a'],src),['a','b']);self.assertEqual(closure(rows,[],src),[])

class RiskReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_attempts_per_pilot_no_expansion(self):
        p={'frames':[dict(event_id=x) for x in ('T30','T08','T29')]}
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)),patch.object(runner,'read',return_value=p),patch.object(runner,'verify'),patch.object(runner.replay,'attempt',new_callable=AsyncMock,return_value={'status':'technical_failure'}) as call:
            original=runner.replay.OUT;await runner.run_replays();self.assertEqual(runner.replay.OUT,original)
            self.assertEqual([(c.args[0]['event_id'],c.args[1]) for c in call.call_args_list],[('T30',1),('T30',2),('T30',3),('T08',1),('T08',2),('T08',3)])

    async def test_semantic_failure_not_retried(self):
        p={'frames':[dict(event_id=x) for x in ('T30','T08','T29')]}
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)),patch.object(runner,'read',return_value=p),patch.object(runner,'verify'),patch.object(runner.replay,'attempt',new_callable=AsyncMock,return_value={'status':'semantic_blocked'}) as call:
            await runner.run_replays();self.assertEqual(call.call_count,2)

    async def test_cancel_restores_output_root(self):
        p={'frames':[dict(event_id='T30')]}
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)),patch.object(runner,'read',return_value=p),patch.object(runner,'verify'),patch.object(runner.replay,'attempt',new_callable=AsyncMock,side_effect=asyncio.CancelledError):
            original=runner.replay.OUT
            with self.assertRaises(asyncio.CancelledError):await runner.run_replays()
            self.assertEqual(runner.replay.OUT,original)

if __name__=='__main__':unittest.main()
