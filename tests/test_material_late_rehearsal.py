import unittest
from collections import Counter
from scripts.vision.material_late_rehearsal_solver import solve

class SolverTests(unittest.TestCase):
    def test_multiset_optimum_determinism(self):
        seq=['a','b','c','a','a','a'];src={x:x for x in 'abc'}
        r=solve(seq,src,7)
        self.assertEqual(Counter(r['members']),Counter(seq))
        self.assertEqual(set(r['members'][-3:]),set('abc'))
        self.assertEqual(r['minimum_changed_slots'],4)
        self.assertEqual(r,solve(seq,src,7))
    def test_already_balanced_and_unresolved(self):
        r=solve(['a','b','a','b'],{'a':'x','b':'y'},17)
        self.assertEqual(r['minimum_changed_slots'],0)
        with self.assertRaises(ValueError):solve(['a'],{},7)

if __name__=='__main__':unittest.main()
