import copy
import unittest
from unittest.mock import patch
from src.ml.artifacts import object_sha256
from scripts.vision.record_material_retention_review import validate
from scripts.vision.material_retention_review_observations import OBS


class ExplicitReviewTests(unittest.TestCase):
    def fixture(self):
        x=dict(event_id='E01',kind='FP',source=dict(image_sha256='im',image_path='impath'),
               page_sha256='page',page_path='pagepath',events=[dict(cell='V-7',seed=7,prediction={'confidence':.5})])
        d=dict(decision_id='E01:0',image_sha256='im',evidence_sha256='page',truth=None,prediction=x['events'][0],
               source_identity=object_sha256(x['source']),event_identity=object_sha256(x['events']),reason='Explicit visual observation',
               reviewed_at='2026-09-11',review_nature='AI-assisted',pixel_visibility_certified=False,
               training_admitted=False,promotable=False,content='cabinet_like',status='observed_not_admitted')
        return {'events':[x]},d

    def test_valid(self):
        e,d=self.fixture()
        with patch('scripts.vision.record_material_retention_review.prior.file_sha256',side_effect=lambda p: {'impath':'im','pagepath':'page'}[p]):
            validate(e,[d])

    def test_missing_duplicate_stale_prediction_source_truth(self):
        e,d=self.fixture()
        for change in ('missing','duplicate','image','page','prediction','source','truth','certification','unknown_pass'):
            x=copy.deepcopy(d);ds=[x]
            if change=='missing':ds=[]
            elif change=='duplicate':ds=[x,x]
            elif change=='image':x['image_sha256']='old'
            elif change=='page':x['evidence_sha256']='old'
            elif change=='prediction':x['prediction']['prediction']['confidence']=.9
            elif change=='source':x['source_identity']='old'
            elif change=='truth':x['truth']={}
            elif change=='certification':x['pixel_visibility_certified']=True
            elif change=='unknown_pass':x['content']='unknown_instance_content'
            with self.subTest(change=change), patch('scripts.vision.record_material_retention_review.prior.file_sha256',side_effect=lambda p:{'impath':'im','pagepath':'page'}[p]):
                with self.assertRaises(ValueError):validate(e,ds)

    def test_complete_explicit_observation_inventory(self):
        self.assertEqual(set(OBS),{f'E{i:02}' for i in range(1,93)})
        self.assertEqual(sum(map(len,OBS.values())),99)
        self.assertEqual(sum(c=='unknown_instance_content' for v in OBS.values() for c,_ in v),0)


if __name__=='__main__':unittest.main()
