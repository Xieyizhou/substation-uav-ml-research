import copy
import unittest
from unittest.mock import patch
from scripts.vision import routed_reflection_control as arm

class ReviewGateTests(unittest.TestCase):
    def records(self):
        return [{},{'positive_decisions':[],'material_decisions':[],'negative_decisions':[]},{},{},
            {'status':'review_complete_not_candidate_passed','integrity':{'integrity_passed':True}},
            {'rows':[{'member_id':'m','page_sha256':'hash'}]},
            {'decisions':[{'member_id':'m','page_sha256':'hash','decision':'bounded_transform_test_allowed','reason':'observed','review_nature':'AI辅助审核'}]}]
    def call(self,rows):
        with patch.object(arm,'checked',side_effect=rows),patch.object(arm,'validate_positive'),patch.object(arm,'validate_review'):
            return arm.review_gate()
    def test_exact(self):self.assertEqual(len(self.call(self.records())),7)
    def test_reject_missing_duplicate_stale_pending_or_nonreview(self):
        for mode in ('missing','duplicate','stale','pending','nonreview','prior_incomplete'):
            rows=self.records();d=rows[-1]['decisions']
            if mode=='missing':d.clear()
            elif mode=='duplicate':d.append(copy.deepcopy(d[0]))
            elif mode=='stale':d[0]['page_sha256']='changed'
            elif mode=='pending':d[0]['decision']='unknown'
            elif mode=='nonreview':d[0]['review_nature']='automatic'
            else:rows[4]['status']='pending'
            with self.assertRaises(ValueError):self.call(rows)

if __name__=='__main__':unittest.main()
