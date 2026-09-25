import copy
import unittest
from collections import Counter
import torch
from scripts.vision.contrast_transfer_control import coefficients,transform,validate_pair,KEYS,VERSION,previous

class ContrastTests(unittest.TestCase):
    def test_coefficients(self):
        for s in (7,17,27):
            self.assertEqual(coefficients(s),coefficients(s))
            self.assertEqual(Counter(coefficients(s)),{.75:960,1.:960,1.25:960})
        self.assertNotEqual(coefficients(7),coefficients(17))
    def test_identity_and_constant(self):
        x=torch.arange(256,dtype=torch.uint8).reshape(1,1,16,16).repeat(1,3,1,1)
        y,c=transform(x,[1.]);self.assertTrue(torch.equal(x,y));self.assertEqual(c,[0])
        for f in (.75,1.,1.25):
            x=torch.full((1,3,4,4),123,dtype=torch.uint8);self.assertTrue(torch.equal(transform(x,[f])[0],x))
    def test_expected_and_no_mutation(self):
        x=torch.tensor([0,100,200,255],dtype=torch.uint8).reshape(1,1,2,2).repeat(1,3,1,1);old=x.clone()
        y,c=transform(x,[1.25]);expected=((x.float()-138.75)*1.25+138.75).round().clamp(0,255).byte()
        self.assertTrue(torch.equal(y,expected));self.assertTrue(torch.equal(x,old));self.assertEqual(c,[6])
        with self.assertRaises(ValueError):transform(x.float(),[1.])
        with self.assertRaises(ValueError):transform(x,[2.])
    def test_frozen_invariants(self):
        old={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')};old['configuration']={'lr':.00025,'augmentation':'off'}
        for f in ('schedules','exposures','windows','listings'):old[f]={k:[s] for k,s in zip(previous.KEYS,(7,17,27))}
        new=copy.deepcopy(old);new['configuration']['augmentation']=VERSION
        for f in ('schedules','exposures','windows','listings'):new[f]={k:old[f][o] for k,o in zip(KEYS,previous.KEYS)}
        new['contrast']={k:coefficients(s) for k,s in zip(KEYS,(7,17,27))};validate_pair(old,new)
        for f in ('pool_rows','training_config','configuration','schedules','contrast'):
            bad=copy.deepcopy(new)
            if f in ('schedules','contrast'):bad[f][KEYS[0]]=[0]
            else:bad[f]=['changed']
            with self.assertRaises(ValueError):validate_pair(old,bad)

if __name__=='__main__':unittest.main()
