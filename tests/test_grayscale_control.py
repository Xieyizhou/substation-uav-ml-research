import unittest
import torch
from scripts.vision.run_grayscale_control import frozen_mask,grayscale,validate_ledger,KEY

class GrayTests(unittest.TestCase):
    def test_mask(self):
        self.assertEqual(frozen_mask(),frozen_mask());self.assertEqual(len(frozen_mask()),600);self.assertEqual(sum(frozen_mask()),300)
    def test_pixels_only_and_formula(self):
        x=torch.rand(6,3,8,8);original=x.clone();y=grayscale(x,[True,False,True,False,False,True])
        self.assertTrue(torch.equal(original,x));self.assertEqual(x.shape,y.shape)
        self.assertTrue(torch.equal(x[1],y[1]));self.assertTrue(torch.equal(y[0,0],y[0,1]));self.assertTrue(torch.equal(y[0,1],y[0,2]))
        self.assertTrue(torch.equal(y[0,0],x[0,0]*.299+x[0,1]*.587+x[0,2]*.114))
    def test_invalid_batch(self):
        with self.assertRaises(ValueError):grayscale(torch.zeros(1,3,8,8),[])
        with self.assertRaises(ValueError):grayscale(torch.zeros(1,1,8,8),[True])
    def test_ledger_rejects_missing_changed_inputs(self):
        p=dict(schedules={KEY:['m']*600},gray_mask=frozen_mask())
        rows=[dict(position=i,member_id='m',grayscale=p['gray_mask'][i],before_sha256='a',after_sha256='a',labels_unchanged=True,geometry_unchanged=True,channels_equal=True) for i in range(600)]
        validate_ledger(dict(rows=rows),p)
        with self.assertRaises(ValueError):validate_ledger(dict(rows=rows[:-1]),p)
        rows[0]['labels_unchanged']=False
        with self.assertRaises(ValueError):validate_ledger(dict(rows=rows),p)

if __name__=='__main__':unittest.main()
