import copy
import unittest
from scripts.vision.contrast_cpu4_control import validate,KEYS,base

class MatchedControlTests(unittest.TestCase):
    def test_only_contrast_changes(self):
        source={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')}
        source['configuration']={'augmentation':'off','lr':.00025,'cpu_threads_per_worker':4}
        for f in ('schedules','exposures','windows','listings'):source[f]={k:[s] for k,s in zip(base.previous.KEYS,(7,17,27))}
        for arm in ('reference','contrast'):
            p=copy.deepcopy(source)
            for f in ('schedules','exposures','windows','listings'):p[f]={k:source[f][o] for k,o in zip(KEYS,base.previous.KEYS)}
            p['configuration']['augmentation']='off' if arm=='reference' else base.VERSION
            p['contrast']={k:([1.]*2880 if arm=='reference' else base.coefficients(s)) for k,s in zip(KEYS,(7,17,27))}
            validate(p,source,arm)
            bad=copy.deepcopy(p);bad['schedules'][KEYS[0]]=[999]
            with self.assertRaises(ValueError):validate(bad,source,arm)
            bad=copy.deepcopy(p);bad['configuration']['lr']=.0005
            with self.assertRaises(ValueError):validate(bad,source,arm)

if __name__=='__main__':unittest.main()
