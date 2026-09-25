import unittest
from collections import Counter
from scripts.vision.freeze_compensated_material_sequences import replace

class Replacement(unittest.TestCase):
    def test_exact_and_deterministic(self):
        old=['old','negative','old','other','old','old']
        args=(old,{'old':3},{'a':2,'b':1},7)
        new,pos=replace(*args)
        self.assertEqual((new,pos),replace(*args))
        self.assertEqual(Counter(new),Counter({'old':1,'negative':1,'other':1,'a':2,'b':1}))
        self.assertEqual(new[1],old[1]);self.assertEqual(new[3],old[3])
    def test_excess_rejected(self):
        with self.assertRaises(ValueError):replace(['x'],{'x':2},{'a':2},7)
    def test_unequal_rejected(self):
        with self.assertRaises(ValueError):replace(['x'],{'x':1},{'a':2},7)

if __name__=='__main__':unittest.main()
