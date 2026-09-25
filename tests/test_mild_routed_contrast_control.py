import copy
import unittest
from scripts.vision.mild_routed_contrast_control import coefficients,transform,validate,KEYS,SOURCE_KEYS,VERSION

class MildTests(unittest.TestCase):
    def test_exact_positions_signs_counts_and_half_amplitude(self):
        old=[.75,1.,1.25]*960;new=coefficients(old)
        self.assertEqual(new,[.875,1.,1.125]*960)
        self.assertEqual([i for i,x in enumerate(old) if x!=1],[i for i,x in enumerate(new) if x!=1])
        self.assertTrue(all(b-1==(a-1)*.5 for a,b in zip(old,new)))
        for bad in (old[:-1],[.8]+old[1:]):
            with self.assertRaises(ValueError):coefficients(bad)
    def test_transform_formula_identity_and_label_independence(self):
        import torch
        x=torch.tensor([0,32,224,255],dtype=torch.uint8).reshape(1,1,2,2).repeat(3,3,1,1)
        out,clipped=transform(x,[.875,1,1.125])
        self.assertTrue(torch.equal(out[1],x[1]))
        expected=((x.float()-127.75)*torch.tensor([.875,1,1.125]).reshape(3,1,1,1)+127.75).round().clamp(0,255).byte()
        self.assertTrue(torch.equal(out,expected));self.assertEqual(clipped,[0,0,6])
        for factors in ([.75,1,1.25],[float('nan'),1,1],[1]):
            with self.assertRaises(ValueError):transform(x,factors)
    def test_non_amplitude_changes_rejected(self):
        old={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')}
        old['pool_rows']=[{'labels':[[1,.1,.2,.3,.4]]}];old['configuration']={'lr':.00025,'augmentation':'routed'}
        for f in ('schedules','exposures','windows','listings'):old[f]={k:['a']*2880 for k in SOURCE_KEYS}
        old['contrast']={k:[.75,1.,1.25]*960 for k in SOURCE_KEYS};p=copy.deepcopy(old)
        for f in ('schedules','exposures','windows','listings'):p[f]={k:['a']*2880 for k in KEYS}
        p['contrast']={k:[.875,1.,1.125]*960 for k in KEYS};p['configuration']['augmentation']=VERSION;validate(old,p)
        for f in ('schedules','exposures','windows','listings','contrast'):
            bad=copy.deepcopy(p);bad[f][KEYS[0]]=[]
            with self.assertRaises(ValueError):validate(old,bad)
        for f in ('pool_rows','initialization','evaluation','environment','held_members','training_config'):
            bad=copy.deepcopy(p);bad[f]=['changed']
            with self.assertRaises(ValueError):validate(old,bad)

if __name__=='__main__':unittest.main()
