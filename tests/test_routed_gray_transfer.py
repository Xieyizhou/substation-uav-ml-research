import copy, unittest
from scripts.vision.routed_gray_transfer_control import transform,validate,KEYS,SOURCE_KEYS,VERSION

class GrayTests(unittest.TestCase):
    def test_identity_and_gray_formula_geometry(self):
        import torch
        from scripts.vision.routed_gray_transfer_control import _contrast
        x=torch.arange(108,dtype=torch.uint8).reshape(3,3,3,4)
        source=x.clone();out,clips=transform(x,[.75,1,1.25]);base,expected_clips=_contrast(x,[.75,1,1.25])
        self.assertTrue(torch.equal(x,source));self.assertTrue(torch.equal(out[1],x[1]));self.assertEqual(out.shape,x.shape)
        for j in (0,2):
            expected=base[j].float().mean(0).round().byte()
            self.assertTrue(all(torch.equal(out[j,c],expected) for c in range(3)))
        self.assertEqual(clips,expected_clips)
    def test_reject_unfrozen_transform(self):
        import torch
        x=torch.zeros((3,3,2,2),dtype=torch.uint8)
        for factors in ([.875,1,1.125],[1],[float('nan'),1,1]):
            with self.assertRaises(ValueError):transform(x,factors)
        with self.assertRaises(ValueError):transform(x.float(),[1,1,1])
        with self.assertRaises(ValueError):transform(x[:,:1],[1,1,1])
    def test_exact_exposures_contrast_and_route(self):
        old={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')}
        old['configuration']={'lr':.00025,'augmentation':'routed'}
        for f in ('schedules','exposures','windows','listings'):old[f]={k:['a']*2880 for k in SOURCE_KEYS}
        old['contrast']={k:[.75,1,1.25]*960 for k in SOURCE_KEYS};p=copy.deepcopy(old)
        for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
        p['configuration']['augmentation']=VERSION;p['grayscale']={k:[True,False,True]*960 for k in KEYS};validate(old,p)
        for f in ('schedules','exposures','windows','listings','contrast','grayscale'):
            bad=copy.deepcopy(p);bad[f][KEYS[0]]=[]
            with self.assertRaises(ValueError):validate(old,bad)
        for f in ('pool_rows','initialization','evaluation','held_members','training_config'):
            bad=copy.deepcopy(p);bad[f]=['changed']
            with self.assertRaises(ValueError):validate(old,bad)

if __name__=='__main__':unittest.main()
