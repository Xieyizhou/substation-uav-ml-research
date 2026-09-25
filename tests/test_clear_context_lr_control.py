import copy
import unittest
from scripts.vision.clear_context_lr_control import validate_pair,training_override,KEYS,LR


class LRControlTests(unittest.TestCase):
    def test_override_only_lr(self):
        config=dict(lr0=.0005,lrf=1.0,epochs=48,batch=6,seed=17)
        result=training_override(config)
        self.assertEqual({k:v for k,v in result.items() if k!='lr0'},{k:v for k,v in config.items() if k!='lr0'})
        self.assertEqual(result['lr0'],LR);self.assertEqual(config['lr0'],.0005)
    def test_drift_rejected(self):
        for config in (dict(lr0=.001,lrf=1.,epochs=48),dict(lr0=.0005,lrf=.1,epochs=48),dict(lr0=.0005,lrf=1.,epochs=96)):
            with self.assertRaises(ValueError):training_override(config)
    def test_pair_invariants(self):
        old={k:[] for k in ('pool_rows','names','initialization','evaluation','environment','held_members')}
        old.update(training_config={'lr0':.0005},configuration={'lr':.0005})
        for f in ('schedules','exposures','windows'):old[f]={f'clear-context-480-{s}':[s] for s in (7,17,27)}
        new=copy.deepcopy(old);new['training_config']['lr0']=LR;new['configuration']['lr']=LR
        for f in ('schedules','exposures','windows'):new[f]={k:[s] for k,s in zip(KEYS,(7,17,27))}
        validate_pair(old,new)
        for field in ('pool_rows','evaluation','schedules'):
            bad=copy.deepcopy(new)
            if field=='schedules':bad[field][KEYS[0]]=[99]
            else:bad[field]=['changed']
            with self.assertRaises(ValueError):validate_pair(old,bad)


if __name__=='__main__':unittest.main()
