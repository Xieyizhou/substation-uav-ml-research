import unittest
from scripts.vision.validate_boundary_roundoff import normalize_box,ulp32

class BoundaryTests(unittest.TestCase):
    def test_observed(self):
        r=normalize_box([0,511.489990234375,606.986572265625,1080.0000305175781])
        self.assertEqual(r['normalized'][-1],1080);self.assertEqual(len(r['changes']),1)
        self.assertEqual(r['raw'][-1],1080.0000305175781)
    def test_inside_unchanged(self):self.assertEqual(normalize_box([1,2,4,5])['changes'],[])
    def test_real_overflow(self):
        with self.assertRaises(ValueError):normalize_box([0,1,5,1080.001])
    def test_beyond_budget(self):
        with self.assertRaises(ValueError):normalize_box([0,1,5,1080+2*ulp32(1080)])
    def test_negative_overflow(self):
        with self.assertRaises(ValueError):normalize_box([-1,1,5,1080])
    def test_nonfinite(self):
        with self.assertRaises(ValueError):normalize_box([0,1,float('nan'),5])
    def test_degenerate(self):
        with self.assertRaises(ValueError):normalize_box([0,0,0,5])
    def test_other_resolution(self):
        with self.assertRaises(ValueError):normalize_box([0,0,3,4],640,640)
    def test_no_mutation(self):
        b=[0,1,5,1080.0000305175781];old=b.copy();normalize_box(b);self.assertEqual(b,old)
