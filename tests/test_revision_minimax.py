from tests.test_revision_compensation import CountTests
from scripts.vision.optimize_revision_compensation import optimize


class MinimaxTests(CountTests):
    def test_minimax_then_L1(self):
        rows=self.rows+[dict(member_id='c',subset='positive',class_instances={'reactor':1})]
        old=dict(self.old,c=5)
        r=optimize(rows,old,{'a'},{'b','c'},{})
        self.assertEqual(r['peak_extra'],3)
        self.assertEqual(r['total_L1'],10)
        self.assertEqual(r['reallocated_slots'],5)
        self.assertEqual(r['counts']['n'],2)
        self.assertEqual(r['counts']['a'],0)

    def test_cap_blocks(self):
        self.assertEqual(optimize(self.rows,self.old,{'a'},set(),{})['status'],'stage1_not_established')
