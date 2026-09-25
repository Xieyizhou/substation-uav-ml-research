import unittest
from scripts.vision.evaluate_reviewed_negative_order import compare_truth
from scripts.vision.finalize_exposure_diagnosis import stats

class ReviewedEvaluationTests(unittest.TestCase):
    def row(self,indices):
        return dict(pair_id='p',view_id='v',variant='original',image_sha256='h',truth=[dict(class_name='reactor',bbox_xyxy=[0,0,10,10]),dict(class_name='reactor',bbox_xyxy=[20,0,30,10])],matches=[dict(truth_index=i) for i in indices])
    def test_gains_losses_by_exact_truth(self):
        self.assertEqual([r['state'] for r in compare_truth(self.row([0]),self.row([1]))],['loss','gain'])
    def test_identity_and_coordinate_conflict(self):
        for field in ('image_sha256','view_id','pair_id','variant','truth'):
            b=self.row([]);b[field]='changed'
            with self.assertRaises(ValueError):compare_truth(self.row([]),b)
    def test_undefined_precision_preserved(self):
        self.assertIsNone(stats([None,None,None])['mean'])
        self.assertEqual(stats([None,.5,None])['defined_seed_count'],1)

if __name__=='__main__':unittest.main()
