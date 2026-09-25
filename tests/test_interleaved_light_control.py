import copy,unittest
from scripts.vision.interleaved_light_control import OUT,REF,SOURCE,prior,constraints,KEYS
class Control(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=prior.read(OUT/'design.json');cls.old=prior.read(SOURCE/'design.json');cls.ref=prior.read(REF/'design.json')
    def test_exact(self):constraints(self.p,self.old,self.ref)
    def test_missing(self):
        p=copy.deepcopy(self.p);p['schedules'][KEYS[0]].pop()
        with self.assertRaises(ValueError):constraints(p,self.old,self.ref)
    def test_reordered(self):
        p=copy.deepcopy(self.p);p['schedules'][KEYS[0]].reverse()
        with self.assertRaises(ValueError):constraints(p,self.old,self.ref)
    def test_brightness(self):
        p=copy.deepcopy(self.p);p['brightness_factors'][KEYS[0]][0]=.123
        with self.assertRaises(ValueError):constraints(p,self.old,self.ref)
    def test_lr(self):
        p=copy.deepcopy(self.p);p['training_config'][KEYS[0]]['lr0']=.001
        with self.assertRaises(ValueError):constraints(p,self.old,self.ref)
    def test_changed_supervision(self):
        p=copy.deepcopy(self.p);m=next(m for m in p['pool_rows'] if m['member_id']==p['schedules'][KEYS[0]][-1] and m['member_id'].startswith('low-')) if p['schedules'][KEYS[0]][-1].startswith('low-') else next(m for m in p['pool_rows'] if m['member_id'].startswith('low-') and m['member_id'] in p['schedules'][KEYS[0]])
        m['class_instances']={'reactor':99}
        with self.assertRaises(ValueError):constraints(p,self.old,self.ref)
if __name__=='__main__':unittest.main()
