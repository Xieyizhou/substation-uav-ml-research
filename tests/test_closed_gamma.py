import unittest
from collections import Counter
import numpy as np
from scripts.vision.closed_gamma_design import schedule
from scripts.vision.closed_gamma_runtime import transform

class GammaTests(unittest.TestCase):
    def test_exact_quotas_windows_and_reproducibility(self):
        for seed in (7,17,27):
            s=schedule(seed);self.assertEqual(s,schedule(seed));self.assertEqual(Counter(s),Counter({1.:450,.8:225,1.2:225}))
            self.assertTrue(all(s[i:i+50].count(1.)==25 for i in range(0,900,50)))
    def test_identity_exact_and_monotonic(self):
        a=np.arange(256,dtype=np.uint8).reshape(1,256,1).repeat(3,axis=2)
        self.assertTrue(np.array_equal(transform(a,1.),a))
        for gamma in (.8,1.2):
            b=transform(a,gamma);self.assertEqual(b.shape,a.shape);self.assertEqual(b.dtype,a.dtype)
            self.assertTrue((np.diff(b[0,:,0].astype(int))>=0).all());self.assertTrue((b[:,0]==0).all());self.assertTrue((b[:,-1]==255).all())
    def test_unfrozen_factor_and_dtype_rejected(self):
        with self.assertRaises(ValueError):transform(np.zeros((2,2,3),dtype=np.uint8),.5)
        with self.assertRaises(ValueError):transform(np.zeros((2,2,3)),1.)

if __name__=='__main__':unittest.main()
