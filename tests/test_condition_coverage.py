import unittest
import xml.etree.ElementTree as ET
from scripts.vision.build_condition_coverage import bbox_features,xml_value

class CoverageTests(unittest.TestCase):
    def test_numeric_xml_equivalence(self):
        self.assertEqual(xml_value(ET.fromstring('<ambient>0.350 0.350 0.350 1</ambient>')),xml_value(ET.fromstring('<ambient>0.35 0.35 0.35 1.0</ambient>')))
    def test_box_serialization_not_relabeling(self):
        r=bbox_features([-9.6e-8,0,96,96],1920,1080)
        self.assertTrue(r['touches_image_edge'])
        with self.assertRaises(ValueError):bbox_features([-.1,0,96,96],1920,1080)
    def test_size_bins(self):
        self.assertEqual(bbox_features([10,10,41,41],640,640)['size_bucket'],'lt32')
        self.assertEqual(bbox_features([10,10,42,42],640,640)['size_bucket'],'32to64')
        self.assertEqual(bbox_features([10,10,74,74],640,640)['size_bucket'],'ge64')
    def test_invalid_box(self):
        with self.assertRaises(ValueError):bbox_features([10,10,10,20],640,640)

if __name__=='__main__':unittest.main()
