import unittest
from copy import deepcopy
from scripts.vision.record_interleaved_error_review import validate_loss
from scripts.vision.build_interleaved_loss_review import transitions


class ReviewTests(unittest.TestCase):
    def evidence(self):
        return dict(objects=[dict(review_id='L01',source={},truth={},events=[{}],page_path='x')],unique_loss_targets=1,new_loss_events=1)

    def test_missing_rejected(self):
        with self.assertRaises(ValueError): validate_loss(self.evidence(),{})

    def test_duplicate_rejected(self):
        e=self.evidence();e['objects']*=2
        with self.assertRaises(ValueError): validate_loss(e,{'L01':('clear_body','seen')})

    def test_unknown_preserved(self):
        d=validate_loss(self.evidence(),{'L01':('unknown','ambiguous')})[0]
        self.assertEqual(d['status'],'pending');self.assertFalse(d['pixel_visibility_certified'])

    def test_count_mismatch_rejected(self):
        e=self.evidence();e['new_loss_events']=2
        with self.assertRaises(ValueError): validate_loss(e,{'L01':('clear_body','seen')})

    def test_identity_not_array_index(self):
        t=[dict(annotation_id=str(i),class_name='reactor',bbox_xyxy=[i,0,i+1,1]) for i in range(2)]
        a=dict(image_sha256='a',pair_id='p',variant='original',truth=t,matches=[dict(truth_index=0)])
        b=dict(a,truth=list(reversed(t)),matches=[dict(truth_index=1)],misses=[dict(truth_index=0)],planned_truth_index=1)
        self.assertEqual([x['transition'] for x in transitions(a,b)],['retained_hit','persistent_miss'])
        b=deepcopy(b);b['truth'][0]['annotation_id']='different'
        with self.assertRaises(ValueError): transitions(a,b)

    def test_pair_conflict(self):
        with self.assertRaises(ValueError): transitions(dict(image_sha256='a'),dict(image_sha256='b'))
