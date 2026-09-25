import unittest
from scripts.vision.export_material_candidate_batch import label_text

class LabelExportTests(unittest.TestCase):
    def test_all_labels_preserved(self):
        r={'objects':[{'class_id':c,'bbox_xyxy':[1,2,101,202]} for c in range(4)]}
        self.assertEqual(len(label_text(r,1920,1080).splitlines()),4)
    def test_outside_rejected(self):
        with self.assertRaises(ValueError):label_text({'objects':[{'class_id':0,'bbox_xyxy':[-1,0,10,10]}]},1920,1080)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):label_text({'objects':[]},1920,1080)
    def test_class_rejected(self):
        with self.assertRaises(ValueError):label_text({'objects':[{'class_id':4,'bbox_xyxy':[0,0,10,10]}]},1920,1080)
