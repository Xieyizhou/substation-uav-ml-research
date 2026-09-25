import unittest
from scripts.vision.reference_batch_order import align_to_reference

class ReferenceOrderTests(unittest.TestCase):
    def test_restores_existing_order_without_member_change(self):
        ref=['z','a','z','b','d','c'];requested=list(reversed(ref))
        self.assertEqual(align_to_reference(requested,ref),ref)
        self.assertEqual(requested,list(reversed(ref)))
    def test_rejects_regrouping_missing_and_collisions(self):
        ref=[str(i) for i in range(12)]
        for requested in (ref[6:]+ref[:6],ref[:-1],['x']+ref[1:],[None]+ref[1:]):
            with self.assertRaises(ValueError):align_to_reference(requested,ref)
