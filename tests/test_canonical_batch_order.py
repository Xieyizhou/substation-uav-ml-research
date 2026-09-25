import unittest,random
from collections import Counter
from scripts.vision.canonical_batch_order import canonicalize

class CanonicalOrderTests(unittest.TestCase):
    def test_permutation_equivalence_and_exposures(self):
        raw=['b','a','a','c','e','d','q','p','n','o','m','l'];permuted=[]
        for i in range(0,len(raw),6):
            b=raw[i:i+6];random.Random(i).shuffle(b);permuted+=b
        self.assertEqual(canonicalize(raw),canonicalize(permuted))
        for i in range(0,len(raw),6):self.assertEqual(Counter(raw[i:i+6]),Counter(canonicalize(raw)[i:i+6]))
        self.assertEqual(canonicalize(canonicalize(raw)),canonicalize(raw))
    def test_invalid_identity_and_partial_batch(self):
        for raw in (['a'],['']*6,[None]*6):
            with self.assertRaises(ValueError):canonicalize(raw)
