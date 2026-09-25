import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.vision.trace_retained_bridge_sources import unique, pixels_equal, check_sync, objects_with_instances

class TraceTests(unittest.TestCase):
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): unique([{'id':'x'},{'id':'x'}],'id')

    def test_pixels_not_container_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.png',Path(folder)/'b.ppm'
            Image.new('RGB',(2,2),'red').save(a);Image.new('RGB',(2,2),'red').save(b)
            self.assertTrue(pixels_equal(a,b))
            Image.new('RGB',(2,2),'blue').save(b)
            self.assertFalse(pixels_equal(a,b))

    def test_sync(self):
        row=dict(rgb_timestamp=1,depth_timestamp=1.01,truth_timestamp=1.02,actual_pose={'timestamp':1.03})
        check_sync(row)
        for value in (1.04,float('nan')):
            row['actual_pose']['timestamp']=value
            with self.assertRaises(ValueError):check_sync(row)

    def test_instances_not_visibility(self):
        truth=dict(image_width=100,image_height=100,objects=[dict(annotation_id='x-instance-0012-box-0',class_name='reactor',bbox_xyxy=[0,2,50,50])])
        mapping={12:dict(object_id='r',category='reactor')}
        obj=objects_with_instances(truth,mapping)[0]
        self.assertTrue(obj['boundary_contact_within_one_pixel'])
        self.assertEqual(obj['visibility_status'],'unknown')
        with self.assertRaises(ValueError):objects_with_instances(truth,{})
        truth['objects']*=2
        with self.assertRaises(ValueError):objects_with_instances(truth,mapping)

if __name__=='__main__':unittest.main()
