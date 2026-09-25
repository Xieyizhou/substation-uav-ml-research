import unittest
from collections import Counter
from scripts.vision.freeze_reviewed_scale_control import factors

class FrozenScaleTests(unittest.TestCase):
    def test_quota_each_window(self):
        f=factors(7,6600);self.assertEqual(len(f),6600)
        for i in range(0,6600,300):self.assertEqual(Counter(f[i:i+300]),Counter({1.:150,.75:75,.5:75}))
    def test_deterministic(self):
        self.assertEqual(factors(17,6600),factors(17,6600));self.assertNotEqual(factors(7,6600),factors(17,6600))
    def test_incomplete(self):
        with self.assertRaises(ValueError):factors(7,299)

if __name__=='__main__':unittest.main()
