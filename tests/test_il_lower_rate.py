import copy,unittest
from scripts.vision.il_lower_rate_control import OUT,REF,KEYS,prior,constraints
class LowerRate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=prior.read(OUT/'design.json');cls.ref=prior.read(REF/'design.json')
    def test_exact(self):constraints(self.p,self.ref)
    def test_lr(self):
        p=copy.deepcopy(self.p);p['training_config'][KEYS[0]]['lr0']=.0005
        with self.assertRaises(ValueError):constraints(p,self.ref)
    def test_steps(self):
        p=copy.deepcopy(self.p);p['training_config'][KEYS[0]]['epochs']=90
        with self.assertRaises(ValueError):constraints(p,self.ref)
    def test_sequence(self):
        p=copy.deepcopy(self.p);p['schedules'][KEYS[0]].reverse()
        with self.assertRaises(ValueError):constraints(p,self.ref)
    def test_augmentation(self):
        p=copy.deepcopy(self.p);p['brightness_factors'][KEYS[0]][0]=.123
        with self.assertRaises(ValueError):constraints(p,self.ref)
    def test_full_supervision(self):
        p=copy.deepcopy(self.p);p['pool_rows'][0]['label_sha256']='invalid'
        with self.assertRaises(ValueError):constraints(p,self.ref)
    def test_threshold(self):
        p=copy.deepcopy(self.p);p['evaluation']['unplanned_threshold']=.3
        with self.assertRaises(ValueError):constraints(p,self.ref)
if __name__=='__main__':unittest.main()
