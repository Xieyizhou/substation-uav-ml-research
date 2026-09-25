import copy
import unittest
from scripts.vision.routed_reflection_control import reflect,positions,validate,KEYS,SOURCE_KEYS,VERSION

class ReflectionTests(unittest.TestCase):
    def test_complete_labels_and_pixels(self):
        import torch
        b=dict(img=torch.arange(108,dtype=torch.uint8).reshape(3,3,3,4),bboxes=torch.tensor([[.2,.4,.3,.5],[.8,.2,.1,.2],[.1,.3,.2,.2]]),batch_idx=torch.tensor([0.,0.,1.]),cls=torch.tensor([[0.],[2.],[1.]]))
        original={k:v.clone() for k,v in b.items()};r=reflect(b,[True,False,True])
        self.assertTrue(torch.equal(r['img'][0],b['img'][0].flip(-1)));self.assertTrue(torch.equal(r['img'][1],b['img'][1]));self.assertTrue(torch.equal(r['img'][2],b['img'][2].flip(-1)))
        self.assertTrue(torch.allclose(r['bboxes'][:,0],torch.tensor([.8,.2,.1])));self.assertTrue(torch.equal(r['bboxes'][:,1:],b['bboxes'][:,1:]))
        for k in ('cls','batch_idx'):self.assertTrue(torch.equal(r[k],b[k]))
        for k in b:self.assertTrue(torch.equal(b[k],original[k]))
        inv=reflect(r,[True,False,True]);self.assertTrue(torch.equal(inv['img'],b['img']));self.assertTrue(torch.allclose(inv['bboxes'],b['bboxes']))
    def test_empty_negative_and_boundary(self):
        import torch
        b=dict(img=torch.zeros((1,3,3,4),dtype=torch.uint8),bboxes=torch.empty((0,4)),batch_idx=torch.empty(0),cls=torch.empty((0,1)))
        self.assertEqual(reflect(b,[True])['bboxes'].shape,(0,4))
        b['bboxes']=torch.tensor([[0.,.5,.1,.1],[1.,.5,.1,.1]]);b['batch_idx']=torch.tensor([0.,0.]);b['cls']=torch.tensor([[1.],[2.]])
        self.assertEqual(reflect(b,[True])['bboxes'][:,0].tolist(),[1.,0.])
        for index in (torch.tensor([.5,0]),torch.tensor([1.,0])):
            with self.assertRaises(ValueError):reflect(dict(b,batch_idx=index),[True])
        with self.assertRaises(ValueError):reflect(b,[1])
        with self.assertRaises(ValueError):reflect(dict(b,bboxes=b['bboxes']*100),[True])
    def test_deterministic_exact_half(self):
        for s in (7,17,27):self.assertEqual(sum(positions(s)),1440);self.assertEqual(positions(s),positions(s))
        self.assertNotEqual(positions(7),positions(17))
    def test_protocol_invariants(self):
        old={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')};old['configuration']={'augmentation':'routed'}
        for f in ('schedules','exposures','windows','listings','contrast'):old[f]={k:['a']*2880 for k in SOURCE_KEYS}
        p=copy.deepcopy(old)
        for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:old[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}
        p['configuration']['augmentation']=VERSION;p['reflection']={k:positions(s) for k,s in zip(KEYS,(7,17,27))};validate(old,p)
        for f in ('schedules','exposures','windows','listings','contrast','reflection'):
            bad=copy.deepcopy(p);bad[f][KEYS[0]]=[]
            with self.assertRaises(ValueError):validate(old,bad)
        bad=copy.deepcopy(p);bad['held_members']=['restore']
        with self.assertRaises(ValueError):validate(old,bad)

if __name__=='__main__':unittest.main()
