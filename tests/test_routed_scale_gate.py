import copy
import unittest
from unittest.mock import patch
from scripts.vision import routed_scale_control as arm

class ScaleGateTests(unittest.TestCase):
    def records(self):
        rows=[dict(member_id=g,group=g,image_sha256='i',label_sha256='l',page_sha256='p') for g in sorted(arm.EXPECTED)]
        decisions=[dict(member_id=r['member_id'],image_sha256='i',label_sha256='l',page_sha256='p',decision='bounded_transform_test_allowed',reason='explicit observation',review_nature='AI辅助审核') for r in rows]
        return dict(rows=rows),dict(decisions=decisions)
    def test_complete(self):
        e,r=self.records()
        with patch.object(arm,'checked',side_effect=[e,r]):self.assertEqual(len(arm.preview_gate()),2)
    def test_missing_duplicate_stale(self):
        e,r=self.records()
        variants=[dict(decisions=r['decisions'][:-1]),dict(decisions=r['decisions'][:-1]+[r['decisions'][0]])]
        for f in ('image_sha256','label_sha256','page_sha256','decision','review_nature','reason'):
            bad=copy.deepcopy(r);bad['decisions'][0][f]='';variants.append(bad)
        for bad in variants:
            with patch.object(arm,'checked',side_effect=[e,bad]):
                with self.assertRaises((ValueError,KeyError)):arm.preview_gate()

if __name__=='__main__':unittest.main()
