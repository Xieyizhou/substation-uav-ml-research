import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter
from scripts.vision import freeze_clear_context_training as f
from scripts.vision import clear_context_training as t
from scripts.vision import review_clear_context_increment as review


class CountsTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        sizes={(0,0,0,1):9,(1,0,0,1):2,(0,0,1,0):6,(1,0,1,0):6,(0,3,0,0):3,(0,2,0,0):2}
        for signature,n in sizes.items():
            for i in range(n):
                mid=f'{signature}-{i}'
                rows.append(dict(member_id=mid,subset='bridge_positive',variant='original',lineage_id=mid,class_instances={c:v for c,v in zip(f.NAMES,signature) if v}))
        rows.append(dict(member_id='negative',subset='hard_negative',variant='original',lineage_id='negative',class_instances={}))
        seq=[r['member_id'] for r in rows[:-1]]*7
        seq+=['negative']*(2880-len(seq))
        signatures=[(1,0,0,0),(1,0,1,0),(1,0,1,0),(0,1,0,0),(0,1,0,1),(0,1,0,0),
                    (0,0,1,0),(0,0,1,0),(0,0,1,0),(0,1,0,1),(1,0,1,1),(0,1,0,1)]
        new=[dict(member_id=f'new-{i}',subset='bridge_positive',variant='original',lineage_id=f'new-pose-{i}',class_instances={c:v for c,v in zip(f.NAMES,s) if v}) for i,s in enumerate(signatures)]
        return rows,new,seq

    def test_exact_budget_instances_and_positions(self):
        rows,new,seq=self.fixture()
        for seed in (7,17,27):
            result,decrements,positions=f.construct(rows,new,seq,seed)
            self.assertEqual(len(positions),120);self.assertEqual(sum(decrements.values()),120)
            self.assertEqual(result,f.construct(rows,new,seq,seed)[0])
            self.assertEqual([i for i,m in enumerate(seq) if m=='negative'],[i for i,m in enumerate(result) if m=='negative'])
            self.assertTrue(all(Counter(result)[r['member_id']]==10 for r in new))
            self.assertTrue(all(Counter(result)[r['member_id']]>=1 for r in rows))

    def test_changed_new_supervision_fails(self):
        rows,new,seq=self.fixture();new[0]['class_instances']['reactor']=1
        with self.assertRaises(ValueError):f.construct(rows,new,seq,7)

    def test_insufficient_capacity_fails(self):
        rows,new,seq=self.fixture();seq=[m if not m.startswith('(1, 0, 0, 1)') else 'negative' for m in seq]
        with self.assertRaises(ValueError):f.construct(rows,new,seq,7)

    def test_negative_position_change_fails(self):
        rows,new,seq=self.fixture();result,_,_=f.construct(rows,new,seq,7)
        result[0],result[-1]=result[-1],result[0]
        with self.assertRaises(ValueError):f.validate_sequence(rows,new,seq,result)

    def test_evaluation_is_chained(self):
        events=[]
        with tempfile.TemporaryDirectory() as directory,patch.object(t,'OUT',Path(directory)),patch.object(t,'launch_gate'),patch.object(t,'checked',return_value={}),patch.object(t,'file_sha256',return_value='test'),patch.object(t,'write_record'),patch.object(t,'parallel',side_effect=lambda flag,folder:events.append(flag)),patch.object(t,'summarize',side_effect=lambda:events.append('summary')):
            t.train()
        self.assertEqual(events,['--worker','--eval-worker','summary'])


class ReviewTests(unittest.TestCase):
    def fixture(self):
        box=dict(annotation_id='a',scene_device_id='device',bbox_xyxy=[1,2,3,4],crop_sha256='crop')
        frame=dict(review_id='C01',image_sha256='image',truth_sha256='truth',page_sha256='page',objects=[box])
        decision={k:v for k,v in frame.items() if k!='objects'}
        decision.update(decision='accepted_for_bounded_training_candidate',full_frame_reason='observed',boxes=[dict(**box,state='visible_identifiable_content',reason='observed')])
        return {'frames':[frame]},{'frames':[decision]}

    def test_complete(self):
        e,r=self.fixture();review.validate(e,r)

    def test_missing_duplicate_stale_unknown_fail(self):
        e,r=self.fixture();variants=[]
        a=copy.deepcopy(r);a['frames']=[];variants.append(a)
        a=copy.deepcopy(r);a['frames']*=2;variants.append(a)
        a=copy.deepcopy(r);a['frames'][0]['page_sha256']='stale';variants.append(a)
        a=copy.deepcopy(r);a['frames'][0]['boxes'][0]['state']='unknown';variants.append(a)
        a=copy.deepcopy(r);a['frames'][0]['boxes'][0]['scene_device_id']='other';variants.append(a)
        for v in variants:
            with self.assertRaises(ValueError):review.validate(e,v)


if __name__=='__main__':unittest.main()
