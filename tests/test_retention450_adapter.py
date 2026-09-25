import unittest
from scripts.vision.prepare_retention450_training import validate_args,KEYS
from scripts.vision.run_matched_appearance_training import ZERO_AUG

class AdapterTests(unittest.TestCase):
    def args(self):
        return dict(imgsz=640,batch=6,nbs=6,device='cpu',workers=0,optimizer='AdamW',lr0=.001,lrf=1.,warmup_epochs=0,epochs=45,amp=False,deterministic=True,val=False,**{k:0 for k in ZERO_AUG})
    def test_450_controls(self):
        validate_args(self.args());self.assertEqual(len(KEYS),6)
    def test_300_rejected(self):
        with self.assertRaises(ValueError):validate_args({**self.args(),'epochs':30})
    def test_augmentation_rejected(self):
        with self.assertRaises(ValueError):validate_args({**self.args(),'mosaic':1})
