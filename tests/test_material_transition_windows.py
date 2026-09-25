import unittest
from scripts.vision.analyze_material_transition_windows import window,candidate_profile


class WindowTests(unittest.TestCase):
    def test_exclusive_start_inclusive_end(self):
        rows={'a':dict(subset='base',class_instances={'x':2},lineage_id='l'),
              'b':dict(subset='negative',class_instances={},lineage_id='n')}
        r=window(['b']*6+['a']*6+['b']*6,rows,1,2,'a')
        self.assertEqual(r['image_count'],6);self.assertEqual(r['class_instances'],{'x':12})
        self.assertEqual(r['target_member_exposures'],6)
        with self.assertRaises(ValueError):window([],rows,2,2,'a')

    def test_candidate_class_and_overlap(self):
        t=dict(class_name='x',bbox_xyxy=[0,0,10,10])
        p=[dict(class_name='x',bbox_xyxy=[0,0,10,10],confidence=.3),
           dict(class_name='y',bbox_xyxy=[0,0,10,10],confidence=.8)]
        r=candidate_profile(t,p)
        self.assertEqual(r['same_class']['confidence'],.3)
        self.assertEqual(r['other_class']['class_name'],'y')
        self.assertIsNone(candidate_profile(t,[])['same_class'])


if __name__=='__main__':unittest.main()
