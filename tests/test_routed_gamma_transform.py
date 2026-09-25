import unittest
import torch
from scripts.vision.routed_gamma_transform import transform,factors

class GammaTest(unittest.TestCase):
    def test_identity_and_non_mutation(self):
        x=torch.arange(256,dtype=torch.uint8).reshape(1,1,16,16).repeat(3,3,1,1);old=x.clone()
        y=transform(x,[.8,1.,1.25])
        self.assertTrue(torch.equal(x,old));self.assertTrue(torch.equal(y[1],x[1]))
        self.assertTrue(bool((y[0]>=x[0]).all()));self.assertTrue(bool((y[2]<=x[2]).all()))
        self.assertEqual(y.shape,x.shape)
        self.assertTrue(bool((y[:,:,:,0][:,:,0]==0).all()))
        self.assertTrue(bool((y[:,:,-1,-1]==255).all()))

    def test_fixed_routing(self):
        self.assertEqual(factors([1.,.75,1.25,1.]),[1.,.8,1.25,1.])
        with self.assertRaises(ValueError):factors([.9])
        with self.assertRaises(ValueError):transform(torch.zeros(1,3,4,4),[1.])
        with self.assertRaises(ValueError):transform(torch.zeros(1,3,4,4,dtype=torch.uint8),[2.])
