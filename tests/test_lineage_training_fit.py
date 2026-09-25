import copy
import unittest
from unittest.mock import patch
from scripts.vision import check_lineage_training_fit as m

class FitTests(unittest.TestCase):
    def test_review_identity_and_full_label(self):
        t={'class_name':'reactor','bbox_xyxy':[0,0,20,20]}
        r={'image_sha256':'a','label_sha256':'b'}
        d={**r,'truth':t}
        m.check_review(d,r,t)
        for key in r:
            bad={**d,key:'changed'}
            with self.assertRaises(ValueError):m.check_review(bad,r,t)
        with self.assertRaises(ValueError):m.check_review(d,r,{**t,'bbox_xyxy':[0,0,21,20]})

    def test_training_prohibited(self):
        with self.assertRaises(RuntimeError):m.forbidden()

    def test_complete_members_and_exposure(self):
        t=[{'class_name':'reactor','bbox_xyxy':[0,0,20,20]}]
        s=m.scoring(t,[],[])
        p={'identity':'p','rows':[{'member_id':'a','image_sha256':'i'}],'models':{'k':{'draws':['a','a']}}}
        u={'status':'complete','model':'k','protocol_identity':'p','optimizer_created':False,'backward_executed':False,'training_validation_executed':False,'rows':[{'member_id':'a','image_sha256':'i','exposures':2,'scoring':s}]}
        with patch.object(m,'verify'),patch.object(m,'truth_for',return_value=t):
            m.valid(u,'k',p)
            bad=copy.deepcopy(u);bad['rows'][0]['exposures']=0
            with self.assertRaises(ValueError):m.valid(bad,'k',p)
            bad=copy.deepcopy(u);bad['rows']*=2
            with self.assertRaises(ValueError):m.valid(bad,'k',p)
            bad=copy.deepcopy(u);bad['optimizer_created']=True
            with self.assertRaises(ValueError):m.valid(bad,'k',p)

    def test_stale_unit_propagates(self):
        with patch.object(m,'verify',side_effect=ValueError('stale')):
            with self.assertRaises(ValueError):m.valid({},'k',{})

    def test_one_to_one_competition(self):
        t=[{'class_name':'reactor','bbox_xyxy':[0,0,20,20]}]*2
        pred=[{'class_name':'reactor','bbox_xyxy':[0,0,20,20],'confidence':.9}]
        s=m.scoring(t,pred,pred)
        self.assertEqual(len(s['matches']),1)
        self.assertEqual(len(s['misses']),1)
        self.assertTrue(s['misses'][0]['formal_matching_competition'])

    def test_publication_preserves_and_extends_inputs(self):
        from scripts.vision import check_lineage_training_fit_v2 as v2
        u={'identity':'old','inputs':{'original':'hash'},'status':'complete'}
        with patch.object(m,'valid'),patch.object(m,'file_sha256',return_value='newhash'),patch.object(m,'frozen',side_effect=lambda path,record:record):
            r=v2.publish(u,'k',{},m.OUT/'result.json')
        self.assertNotIn('identity',r)
        self.assertEqual(r['inputs']['original'],'hash')
        self.assertEqual(len(r['inputs']),3)

if __name__=='__main__':unittest.main()
