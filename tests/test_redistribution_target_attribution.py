import unittest
from scripts.vision.audit_redistribution_targets import resolve

class TargetAttributionTests(unittest.TestCase):
    def setUp(self):
        self.raw={'label':178,'box':{'minCorner':{'x':1,'y':2},'maxCorner':{'x':10,'y':20}}}
        self.truth={'bbox_xyxy':[1,2,10,20],'class_name':'capacitor_bank'}
        self.mapping={178:{'object_id':'capacitor_east','category':'capacitor_bank'}}
    def test_unique_target(self):self.assertEqual(resolve([self.raw],self.truth,self.mapping)[0],178)
    def test_duplicate_or_missing(self):
        for rows in [[],[self.raw,self.raw]]:
            with self.assertRaises(ValueError):resolve(rows,self.truth,self.mapping)
    def test_mapping_or_category_missing(self):
        for mapping in [{},{178:{'object_id':'x','category':'transformer'}}]:
            with self.assertRaises(ValueError):resolve([self.raw],self.truth,mapping)
    def test_coordinates_changed(self):
        with self.assertRaises(ValueError):resolve([self.raw],{**self.truth,'bbox_xyxy':[1,2,11,20]},self.mapping)
