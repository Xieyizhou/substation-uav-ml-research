import unittest
import numpy as np
from ultralytics.utils.instance import Instances
from scripts.vision.reviewed_scale_runtime import Scale

class RuntimeTests(unittest.TestCase):
    def labels(self):return dict(img=np.full((80,100,3),40,dtype=np.uint8),cls=np.array([[3],[1]],dtype=np.float32),instances=Instances(np.array([[.5,.5,1,1],[.2,.3,.2,.2]],dtype=np.float32),segments=np.zeros((0,0,2)),bbox_format='xywh',normalized=True))
    def test_full_geometry(self):
        labels=self.labels();log=[{}];out=Scale(dict(scale_factors={'k':[.5]}),'k',log)(labels)
        np.testing.assert_allclose(out['instances'].bboxes,[[25,20,75,60],[30,28,40,36]],atol=1e-5)
        self.assertEqual(len(out['cls']),2);self.assertEqual(out['img'][0,0,0],114)
    def test_identity_no_mutation(self):
        labels=self.labels();before=labels['img'].copy();boxes=labels['instances'].bboxes.copy();log=[{}]
        Scale(dict(scale_factors={'k':[1.]}),'k',log)(labels)
        np.testing.assert_array_equal(before,labels['img']);np.testing.assert_array_equal(boxes,labels['instances'].bboxes)
        self.assertTrue(labels['instances'].normalized)
    def test_sequence_and_empty(self):
        with self.assertRaises(ValueError):Scale(dict(scale_factors={'k':[1.]}),'k',[])(self.labels())
        labels=self.labels();labels['cls']=np.zeros((0,1),dtype=np.float32);labels['instances']=Instances(np.zeros((0,4),dtype=np.float32),segments=np.zeros((0,0,2)),bbox_format='xywh',normalized=True)
        out=Scale(dict(scale_factors={'k':[.75]}),'k',[{}])(labels);self.assertEqual(len(out['instances']),0)

if __name__=='__main__':unittest.main()
