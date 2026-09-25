import unittest
from unittest.mock import patch
import numpy as np
from scripts.vision.brightness_transfer_runtime import brighten,factor,Brightness,check_log

class BrightnessTests(unittest.TestCase):
    def test_bounds_and_determinism(self):
        a=[factor(7,'member',i) for i in range(2700)]
        self.assertEqual(a,[factor(7,'member',i) for i in range(2700)])
        self.assertTrue(all(.8<=x<=1.2 for x in a));self.assertTrue(any(x==1 for x in a));self.assertTrue(any(x!=1 for x in a))
    def test_identity_exact(self):
        im=np.arange(3*10*10,dtype=np.uint8).reshape(10,10,3)
        out=brighten(im,1);np.testing.assert_array_equal(im,out);self.assertIsNot(im,out)
    def test_rng_not_consumed(self):
        im=np.full((10,10,3),80,np.uint8)
        with patch.object(np.random,'uniform',side_effect=AssertionError('global RNG')):
            self.assertEqual(brighten(im,.8)[0,0,0],64);factor(7,'m',0)
    def test_matches_value_only_HSV(self):
        import cv2
        im=np.array([[[40,80,120],[0,255,255],[255,255,255],[0,0,0]]],dtype=np.uint8)
        for gain in (.8,1.2):
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);hsv[:,:,2]=np.clip(hsv[:,:,2].astype(float)*gain,0,255).astype(np.uint8)
            np.testing.assert_array_equal(brighten(im,gain),cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
    def test_labels_preserved_and_position_checked(self):
        p={'pool_rows':[{'image_path':'i','member_id':'m'}],'schedules':{'k':['m']},'brightness_factors':{'k':[.8]}}
        log=[];t=Brightness(p,'k',log);labels={'im_file':'i','img':np.full((2,2,3),100,np.uint8),'cls':np.array([1]),'boxes':np.array([[0,0,1,1]])}
        old=labels['boxes'].copy();t(labels);np.testing.assert_array_equal(labels['boxes'],old);self.assertEqual(log[0]['gain'],.8)
        with self.assertRaises(IndexError):t(labels)
    def test_incomplete_ledger_fails(self):
        with self.assertRaises(ValueError):check_log({},'k',[])
    def test_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError):brighten(np.zeros((1,1,3),np.uint8),.5)

if __name__=='__main__':unittest.main()
