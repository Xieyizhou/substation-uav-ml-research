import copy
import unittest
from unittest.mock import patch
from scripts.vision.diagnose_closed_pool_fit import validate
from scripts.vision.structure_fit import scoring
from scripts.vision.summarize_closed_pool_fit import metrics

class PoolFitTests(unittest.TestCase):
    def test_full_truth_and_exposure_validation(self):
        t=[dict(class_name='reactor',bbox_xyxy=[0,0,20,20])]
        m=dict(member_id='m',truth=t)
        p=dict(identity='p',members=[m],exposures={'M-7':{'m':2}})
        r=dict(protocol_identity='p',model='M-7',rows=[dict(member_id='m',exposures=2,**scoring(t,[],[]))])
        with patch('scripts.vision.diagnose_closed_pool_fit.prior.verify'):
            validate(r,p,'M-7')
            for field,value in (('exposures',0),('member_id','other'),('truth',[])):
                changed=copy.deepcopy(r);changed['rows'][0][field]=value
                with self.assertRaises(ValueError):validate(changed,p,'M-7')
    def test_zero_denominator_and_negative_counts(self):
        self.assertEqual(metrics([])['recall'],{})
        r=scoring([],[],[])
        self.assertEqual(metrics([r])['negative_images'],1)
        self.assertEqual(metrics([r])['negative_false_positive_images'],0)
    def test_one_to_one_competition(self):
        t=[dict(class_name='reactor',bbox_xyxy=[0,0,20,20])]*2
        pred=[dict(class_name='reactor',bbox_xyxy=[0,0,20,20],confidence=.8)]
        r=scoring(t,pred,pred)
        self.assertEqual(len(r['matches']),1)
        self.assertTrue(r['misses'][0]['formal_matching_competition'])
        self.assertEqual(metrics([r])['recall']['reactor'],.5)

if __name__=='__main__':unittest.main()
