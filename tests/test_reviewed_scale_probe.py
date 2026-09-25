import unittest
from PIL import Image
from scripts.vision.probe_reviewed_scale import transform

class ScaleProbeTests(unittest.TestCase):
    def test_identity(self):
        im=Image.new('RGB',(100,80),(9,30,200));truth=[dict(class_name='reactor',bbox_xyxy=[1,2,99,79])]
        out,t=transform(im,truth,1.);self.assertEqual(out.tobytes(),im.tobytes());self.assertEqual(t,truth)
    def test_full_labels_no_clipping(self):
        im=Image.new('RGB',(100,80));truth=[dict(class_name='reactor',bbox_xyxy=[0,0,100,80]),dict(class_name='switchgear',bbox_xyxy=[10,20,30,40])]
        out,t=transform(im,truth,.5);self.assertEqual(out.size,im.size);self.assertEqual(len(t),2)
        self.assertEqual(t[0]['bbox_xyxy'],[25,20,75,60]);self.assertEqual(t[1]['bbox_xyxy'],[30,30,40,40]);self.assertEqual(truth[0]['bbox_xyxy'],[0,0,100,80])
    def test_invalid(self):
        with self.assertRaises(ValueError):transform(Image.new('RGB',(20,20)),[],.9)
        with self.assertRaises(ValueError):transform(Image.new('RGB',(20,20)),[dict(bbox_xyxy=[-1,0,2,2])],.5)
    def test_subpixel_roundoff_preserved(self):
        truth=[dict(bbox_xyxy=[-9.6e-8,0,2,2])]
        _,result=transform(Image.new('RGB',(20,20)),truth,1.)
        for actual,expected in zip(result[0]['bbox_xyxy'],truth[0]['bbox_xyxy']):
            self.assertAlmostEqual(actual,expected,delta=1e-15)

if __name__=='__main__':unittest.main()
