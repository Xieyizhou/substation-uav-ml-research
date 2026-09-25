import unittest
import numpy as np
from scripts.vision.audit_material_mask_coverage import coverage

class MaskCoverageTests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[[1,0,76],[1,0,118]],[[0,0,0],[1,0,118]]],dtype='u1')
        self.m={'76':{'object_id':'transformer_sw'},'118':{'object_id':'cabinet'}}
    def test_unboxed_instance(self):
        r=coverage(self.a,[118],self.m)
        self.assertEqual(r['status'],'blocked_unboxed_target_pixels')
        self.assertEqual(r['missing_targets'][0]['visible_pixels'],1)
    def test_all_instances_boxed(self):self.assertEqual(coverage(self.a,[76,118],self.m)['missing_targets'],[])
    def test_collision(self):
        with self.assertRaises(ValueError):coverage(self.a,[76,118,118],self.m)
    def test_unknown_id(self):
        with self.assertRaises(ValueError):coverage(self.a,[118],{'118':self.m['118']})
    def test_category_mask(self):
        self.a[:,:,:2]=0
        with self.assertRaises(ValueError):coverage(self.a,[76,118],self.m)
    def test_empty_mask_unknown(self):
        with self.assertRaises(ValueError):coverage(np.zeros((2,2,3),dtype='u1'),[],self.m)
