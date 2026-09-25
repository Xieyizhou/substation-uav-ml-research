import copy
import unittest
from unittest.mock import patch
from scripts.vision import routed_gamma_control as a

class GammaContractTest(unittest.TestCase):
    def test_frozen_contract_rejects_changes(self):
        s=a.checked(a.SOURCE/'protocol.json');p=a.checked(a.OUT/'protocol.json');a.validate(s,p)
        for field in ('schedules','contrast','gamma_factors'):
            bad=copy.deepcopy(p);bad[field][a.KEYS[0]][0]='invalid'
            with self.assertRaises(ValueError):a.validate(s,bad)
        bad=copy.deepcopy(p);bad['pool_rows'][0]['label_sha256']='invalid'
        with self.assertRaises(ValueError):a.validate(s,bad)

    def test_review_missing_duplicate_stale(self):
        e=a.checked(a.OUT/'preview-v1/evidence.json');r=a.checked(a.OUT/'preview-v1/review.json')
        a.preview_gate()
        for kind in ('missing','duplicate','stale'):
            bad=copy.deepcopy(r)
            if kind=='missing':bad['decisions'].pop()
            elif kind=='duplicate':bad['decisions'].append(bad['decisions'][0])
            else:bad['decisions'][0]['page_sha256']='invalid'
            with patch.object(a,'checked',side_effect=[e,bad]),self.assertRaises(ValueError):a.preview_gate()

    def test_nonmaterial_identity_and_bounded_factors(self):
        p=a.checked(a.OUT/'protocol.json');by={r['member_id']:r for r in p['pool_rows']}
        for key in a.KEYS:
            for member,g in zip(p['schedules'][key],p['gamma_factors'][key]):
                self.assertIn(g,(.8,1.,1.25))
                if by[member]['variant'] not in a.routed.MATERIAL:self.assertEqual(g,1.)

if __name__=='__main__':unittest.main()
