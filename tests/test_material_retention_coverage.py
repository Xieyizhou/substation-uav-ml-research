import copy
import unittest
from collections import Counter
from scripts.vision.freeze_material_retention_coverage import freeze,checks,source,prior,allocate
from scripts.vision.resolve_material_scope_duplicates import inspect


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=freeze();cls.old=prior.read(source.OUT/'protocol.json')

    def test_real_counts(self):
        checks(self.p,self.old)
        for seed in (7,17,27):
            self.assertEqual(Counter(allocate(self.p['treatment_positions'][str(seed)],seed).values()),Counter(warm=10,cool=10,gray=10))

    def test_schedule_brightness_scope_and_allocation_tampering(self):
        for change in ('length','brightness','untreated','scope','allocation'):
            p=copy.deepcopy(self.p)
            if change=='length':p['schedules']['T-7'].pop()
            elif change=='brightness':p['brightness_factors']['A-7'][0]=.123456
            elif change=='untreated':
                affected={r['position'] for r in p['treatment_positions']['7']}
                i=next(i for i in range(2700) if i not in affected);p['schedules']['T-7'][i]='G01-gray_target_body'
            elif change=='scope':
                i=next(int(k) for k,v in p['allocation']['7'].items() if v=='gray');p['schedules']['A-7'][i]=p['schedules']['T-7'][i]
            else:p['allocation']['7'][next(iter(p['allocation']['7']))]='invalid'
            with self.subTest(change=change),self.assertRaises(ValueError):checks(p,self.old)

    def test_external_overlap_not_waived(self):
        with self.assertRaises(ValueError):inspect(dict(members=[{'member_id':'m'}],reference_overlap_gaps=[dict(member_id='m',overlap=['paired_review'])]))

    def test_class_quota_infeasible_rejected(self):
        x=copy.deepcopy(self.p['treatment_positions']['7']);x.pop()
        with self.assertRaises(ValueError):allocate(x,7)


if __name__=='__main__':unittest.main()
