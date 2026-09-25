import unittest
from scripts.vision.small_scale_interleave import permutation, reorder, validate


class Interleave(unittest.TestCase):
    def setUp(self):
        self.seq = list(range(6600)); self.gains = [i/10000 for i in self.seq]
        self.order = permutation()

    def test_order(self):
        self.assertEqual(sorted(self.order), list(range(1100)))
        self.assertEqual([i for i in self.order if i < 1000], list(range(1000)))
        self.assertEqual([i for i in self.order if i >= 1000], list(range(1000,1100)))
        self.assertEqual([i for i,b in enumerate(self.order) if b >= 1000], list(range(10,1100,11)))

    def test_exact(self):
        self.assertTrue(validate(self.seq,self.gains,reorder(self.seq,self.order),reorder(self.gains,self.order),self.order))

    def test_duplicate_missing(self):
        for order in (self.order[:-1], [0]*1100):
            with self.assertRaises(ValueError): reorder(self.seq,order)

    def test_member_or_gain_drift(self):
        seq = reorder(self.seq,self.order); gains = reorder(self.gains,self.order)
        seq[0],seq[1] = seq[1],seq[0]
        with self.assertRaises(ValueError): validate(self.seq,self.gains,seq,gains,self.order)
        with self.assertRaises(ValueError): validate(self.seq,self.gains,reorder(self.seq,self.order),self.gains,self.order)


if __name__ == '__main__': unittest.main()
