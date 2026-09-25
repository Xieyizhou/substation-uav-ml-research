import unittest
from scripts.vision.summarize_material_late_rehearsal_fit import match_truth

class FitIdentityTests(unittest.TestCase):
    def test_reordered_full_truth(self):
        a=dict(object_id='a',class_name='x',bbox_xyxy=[0,0,1,1]);b=dict(object_id='b',class_name='x',bbox_xyxy=[2,2,3,3])
        self.assertEqual(match_truth([a,b],[b,a]),[(0,1),(1,0)])
        with self.assertRaises(ValueError):match_truth([a,a],[a,a])
        with self.assertRaises(ValueError):match_truth([a],[b])

if __name__=='__main__':unittest.main()
