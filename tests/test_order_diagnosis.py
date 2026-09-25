import unittest,copy
from scripts.vision.run_order_diagnosis import transform,validate_transforms,batches,KEYS

class OrderTests(unittest.TestCase):
    def setUp(self):self.r=[str(i%222) for i in range(600)];self.s=transform(self.r)
    def test_deterministic_and_constraints(self):
        self.assertEqual(self.s,transform(self.r));validate_transforms(self.r,self.s)
    def test_changed_exposure_rejected(self):
        s=copy.deepcopy(self.s);s[KEYS[0]][0]='new'
        with self.assertRaises(ValueError):validate_transforms(self.r,s)
    def test_wrong_batch_membership_rejected(self):
        s=copy.deepcopy(self.s);s[KEYS[0]][0],s[KEYS[0]][6]=s[KEYS[0]][6],s[KEYS[0]][0]
        with self.assertRaises(ValueError):validate_transforms(self.r,s)
    def test_wrong_internal_order_rejected(self):
        s=copy.deepcopy(self.s);s[KEYS[1]][0],s[KEYS[1]][1]=s[KEYS[1]][1],s[KEYS[1]][0]
        with self.assertRaises(ValueError):validate_transforms(self.r,s)
