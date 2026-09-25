import unittest
import torch
from scripts.vision.routed_scale_transform import factors,transform

class ScaleTests(unittest.TestCase):
    def batch(self):
        return dict(img=torch.zeros((2,3,640,640),dtype=torch.uint8),bboxes=torch.tensor([[.5,.5,1.,1.],[.25,.25,.1,.2],[.5,.5,.4,.4]]),batch_idx=torch.tensor([0.,0.,1.]),cls=torch.tensor([[0.],[1.],[2.]]))
    def test_identity_and_no_source_mutation(self):
        b=self.batch();r=transform(b,[1.,1.])
        for k in b:self.assertTrue(torch.equal(b[k],r[k]))
        transform(b,[.75,1.]);self.assertEqual(b['img'].sum().item(),0)
    def test_all_labels_and_boundary(self):
        b=self.batch();r=transform(b,[.75,1.])
        self.assertTrue(torch.equal(r['bboxes'][0],torch.tensor([.5,.5,.75,.75])))
        self.assertTrue(torch.allclose(r['bboxes'][1],torch.tensor([.3125,.3125,.075,.15])))
        self.assertTrue(torch.equal(r['bboxes'][2],b['bboxes'][2]))
        for k in ('cls','batch_idx'):self.assertTrue(torch.equal(b[k],r[k]))
        self.assertTrue((r['img'][0,:,:80]==114).all());self.assertTrue((r['img'][0,:,80:560,80:560]==0).all())
    def test_empty_and_invalid(self):
        b=self.batch();b.update(bboxes=torch.empty(0,4),cls=torch.empty(0,1),batch_idx=torch.empty(0));self.assertEqual(len(transform(b,[.75,1.])['bboxes']),0)
        with self.assertRaises(ValueError):transform(b,[.5,1.])
        b=self.batch();b['batch_idx'][0]=2
        with self.assertRaises(ValueError):transform(b,[.75,1.])
    def test_deterministic_routing(self):
        rows=[dict(member_id='a',variant='neutral'),dict(member_id='b',variant='original')];seq=['a','b']*11
        x=factors(rows,seq,7);self.assertEqual(x,factors(rows,seq,7));self.assertEqual(x.count(.75),5);self.assertTrue(all(x[i]==1 for i in range(1,22,2)))
        with self.assertRaises(ValueError):factors(rows,['missing'],7)

if __name__=='__main__':unittest.main()
