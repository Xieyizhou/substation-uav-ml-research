import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from scripts.vision.freeze_full_image_training import encode, check_source, schedules, VARIANTS
from scripts.vision.exposure_protocol import save, verify, read
from src.ml.artifacts import object_sha256, file_sha256

class FreezeTests(unittest.TestCase):
    def obj(self, **updates):
        return dict(dict(object_id='target',category='transformer',runtime_label=1,
            bbox_xyxy=[10.,20.,60.,80.],review_status='visible_content_observed',reason='visible body',planned=True),**updates)

    def test_full_frame_not_target_only(self):
        rows=encode([self.obj(),self.obj(object_id='other',category='reactor',planned=False)],(100,100)).splitlines()
        self.assertEqual(len(rows),2);self.assertTrue(rows[1].startswith('3 '))

    def test_invalid_boxes_are_not_clipped(self):
        for box in ([-1,0,10,10],[0,0,101,10],[1,1,1,2],[0,0,float('nan'),2]):
            with self.assertRaises(ValueError):encode([self.obj(bbox_xyxy=box)],(100,100))

    def test_missing_review_duplicate_unknown_class(self):
        for objects in ([self.obj(review_status='pending')],[self.obj(reason='')],
                        [self.obj(),self.obj()],[self.obj(category='unknown')],[]):
            with self.assertRaises(ValueError):encode(objects,(100,100))

    def test_raw_truth_and_nonplanned_identity(self):
        truth={'annotatedBox':[dict(label=1,box=dict(minCorner=dict(x=10,y=20),maxCorner=dict(x=60,y=80)))]}
        frame=dict(view_id='v',image_sha256='i',truth_sha256=object_sha256(truth),objects=[self.obj()])
        row=dict(status='captured',view_id='v',image_sha256='i',raw_truth=truth)
        mapping={'1':dict(object_id='target',category='transformer')}
        check_source(frame,row,mapping)
        frame['objects']=[]
        with self.assertRaises(ValueError):check_source(frame,row,mapping)
        frame['objects']=[self.obj()];frame['truth_sha256']='stale'
        with self.assertRaises(ValueError):check_source(frame,row,mapping)

    def fixture(self):
        rows=[dict(member_id=s,subset=s,class_instances={'transformer':1}) for s in ('base','regular','bridge_positive','hard_negative')]
        block=[s for s,n in [('base',216),('regular',156),('bridge_positive',120),('hard_negative',108)] for _ in range(n)]
        prior=dict(pool_rows=rows,schedules={f'I-300-{seed}':block*3 for seed in (7,17,27)})
        members=[dict(member_id=f'{p}:{v}',pair_id=p,variant=v,class_instances={'transformer':1})
                 for p in [f'F{i:02}' for i in range(1,9)] for v in ('original',)+VARIANTS]
        return prior,members

    def test_exact_paired_exposure_and_balance(self):
        prior,members=self.fixture()
        for seed in (7,17,27):
            a,b,swaps=schedules(prior,members,seed)
            self.assertEqual(len(a),1800);self.assertEqual(len(swaps),360)
            self.assertEqual(Counter(s['pair_id'] for s in swaps),{f'F{i:02}':45 for i in range(1,9)})
            for p in {s['pair_id'] for s in swaps}:
                c=Counter(s['variant'] for s in swaps if s['pair_id']==p)
                self.assertLessEqual(max(c.values())-min(c.values()),1)
            for start in (0,600,1200):
                self.assertEqual(sum(x!=y for x,y in zip(a[start:start+600],b[start:start+600])),120)
                self.assertEqual(Counter(x for x in a[start:start+600] if ':' not in x),{'base':216,'regular':156,'hard_negative':108})
            self.assertEqual((a,b,swaps),schedules(prior,members,seed))

    def test_class_supervision_mismatch_rejected(self):
        prior,members=self.fixture();members[1]['class_instances']={'reactor':1}
        with self.assertRaises(ValueError):schedules(prior,members,7)

    def test_missing_variant_rejected(self):
        prior,members=self.fixture()
        with self.assertRaises(KeyError):schedules(prior,members[:-1],7)

    def test_stale_export_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'label.txt';p.write_text('original')
            r=save(Path(d)/'receipt.json',dict(inputs={str(p):file_sha256(p)}))
            verify(r);p.write_text('changed')
            with self.assertRaises(ValueError):verify(r)
            p.unlink()
            with self.assertRaises(FileNotFoundError):verify(r)

if __name__=='__main__':unittest.main()
