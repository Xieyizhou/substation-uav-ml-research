"""Fail-closed review checks independent of real experiment data."""
import copy, unittest
from unittest.mock import patch
from scripts.vision import routed_gray_transfer_control as arm

class GateTests(unittest.TestCase):
    def records(self):
        p={'rows':[{'member_id':'a','page_sha256':'x'}]}
        r={'decisions':[{'member_id':'a','page_sha256':'x','decision':'bounded_transform_test_allowed','reason':'observed'}]}
        return [{},{'positive_decisions':[],'negative_decisions':[]},{},{},{'positive_decisions':[]},{'integrity':{'integrity_passed':True}},p,r]
    def call(self,records):
        with patch.object(arm,'checked',side_effect=records),patch.object(arm,'validate_positive'),patch.object(arm,'validate_review'):
            return arm.review_gate()
    def test_exact_preview(self):self.assertEqual(len(self.call(self.records())),8)
    def test_missing_duplicate_stale_or_pending(self):
        for kind in ('missing','duplicate','stale','pending'):
            rows=self.records();r=rows[-1]['decisions']
            if kind=='missing':r.clear()
            elif kind=='duplicate':r.append(copy.deepcopy(r[0]))
            elif kind=='stale':r[0]['page_sha256']='wrong'
            else:r[0]['decision']='unknown'
            with self.assertRaises(ValueError):self.call(rows)

if __name__=='__main__':unittest.main()
