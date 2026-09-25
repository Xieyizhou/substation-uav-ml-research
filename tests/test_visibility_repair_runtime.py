import unittest
from scripts.vision.run_visibility_repair_training import verify_indices

class RuntimeTests(unittest.TestCase):
    def test_exact_order(self):verify_indices(['a','b','a'],['a','b','a'])
    def test_wrong_order_or_missing(self):
        for actual in (['a','a','b'],['a','b'],['a','b','a','a']):
            with self.assertRaises(ValueError):verify_indices(['a','b','a'],actual)

if __name__=='__main__':unittest.main()
