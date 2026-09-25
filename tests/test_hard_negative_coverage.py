import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.vision.hard_negative_coverage import COUNTS,PILOT,projected
from scripts.vision.hard_negative_coverage_admission import normalized_world,validate_decisions
from scripts.vision.train_hard_negative_coverage import expanded_draws,checked_cell
from scripts.vision.exposure_protocol import save
from collections import Counter

class CoverageTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(sum(COUNTS.values()),48)
        self.assertEqual(sum(PILOT.values()),12)
        self.assertTrue(all(PILOT[k]<=COUNTS[k] for k in COUNTS))

    def test_world_diff(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.sdf',Path(d)/'b.sdf'
            a.write_text('<sdf><world><light name="sun"><diffuse>1 1 1 1</diffuse></light><model name="cabinet"/></world></sdf>')
            b.write_text('<sdf><world><light name="sun"><diffuse>.4 .5 .6 1</diffuse></light><model name="cabinet"/><scene><ambient>.3 .3 .4 1</ambient></scene></world></sdf>')
            self.assertEqual(normalized_world(a),normalized_world(b))
            b.write_text(b.read_text().replace('name="cabinet"','name="other"'))
            self.assertNotEqual(normalized_world(a),normalized_world(b))

    def test_missing_stale_and_unknown_review(self):
        frame=dict(view_id='a',image_path='unused',image_sha256='abc')
        decision=dict(view_id='a',image_sha256='abc',decision='accepted',no_target_visible=True,
            coverage_confirmed=True,review_nature='AI-assisted',reason='looked',reviewed_at='now',
            rois=[dict(content='building',bbox_xyxy=[0,0,10,10])])
        with patch('scripts.vision.hard_negative_coverage_admission.file_sha256',return_value='abc'):
            validate_decisions([frame],[decision])
            for values in ([],[decision,decision],[{**decision,'image_sha256':'stale'}],
                           [{**decision,'coverage_confirmed':False}],
                           [{**decision,'rois':[dict(content='unknown',bbox_xyxy=[0,0,10,10])]}]):
                with self.assertRaises(ValueError):validate_decisions([frame],values)

    def test_projection(self):
        row=dict(camera_position=[0,0,1],orientation=[0,0,0,1])
        box=projected(row,[5,6,-1,1,0,2])
        self.assertTrue(0<box[0]<box[2]<1)
        self.assertIsNone(projected(row,[-5,-4,-1,1,0,2]))

    def test_negative_composition_only(self):
        negatives=[dict(member_id=f'n{i}',lineage_id=f'p{i//2}',subset='hard_negative') for i in range(120)]
        positives=[dict(member_id='positive',subset='base')]
        old=['positive']*492+[f'n{i%24}' for i in range(108)]
        new,omitted=expanded_draws(positives+negatives[:24],negatives[24:],7,old)
        self.assertEqual(len(new),600);self.assertEqual(new[:492],old[:492])
        self.assertEqual(len(set(new[492:])),108);self.assertEqual(len(omitted),6)
        self.assertEqual(set(Counter(int(mid[1:])//2 for mid in new[492:]).values()),{2})
        self.assertEqual((new,omitted),expanded_draws(positives+negatives[:24],negatives[24:],7,old))
        with self.assertRaises(ValueError):expanded_draws(positives+negatives[:24],negatives[25:],7,old)

    def test_resumed_actual_exposure_cannot_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);actual=root/'exposure.json';completion=root/'completion.json'
            protocol=dict(identity='frozen',schedules={'O-100-7':['member']},exposures={'O-100-7':{'draws':1}})
            save(actual,dict(draws=['member'],summary={'draws':1}))
            save(completion,dict(status='complete',protocol_identity='frozen',cell='O-100-7',exposure_path=str(actual),optimizer_steps=100))
            with patch('scripts.vision.train_hard_negative_coverage.verify_tree'):
                checked_cell(completion,protocol)
                save(actual,dict(draws=['different-member'],summary={'draws':1}))
                with self.assertRaises(ValueError):checked_cell(completion,protocol)

if __name__=='__main__':unittest.main()
