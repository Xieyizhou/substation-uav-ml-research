import unittest
from scripts.vision.prove_material_pose_budget_conflict import witness

class BudgetConflict(unittest.TestCase):
    def test_exact_deficit(self):
        r=witness({'old':{'reactor':1}}, {'a':{'switchgear':6},'b':{'capacitor_bank':1}})
        self.assertEqual(r['minimum_deficit_all_poses_once'],1)
        self.assertEqual(r['minimum_deficit_all_three_appearances'],3)
    def test_compensating_vector_invalidates_proof(self):
        with self.assertRaises(ValueError):witness({'old':{'reactor':1}}, {'a':{},'b':{'capacitor_bank':1,'reactor':1}})
    def test_old_invariant_required(self):
        with self.assertRaises(ValueError):witness({'old':{}}, {'a':{}})
    def test_no_deficit_not_infeasible(self):
        with self.assertRaises(ValueError):witness({'old':{'reactor':1}}, {'a':{'capacitor_bank':1}})

if __name__=='__main__':unittest.main()
