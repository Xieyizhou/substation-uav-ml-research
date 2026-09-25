import copy
import unittest
from collections import Counter
from scripts.vision.dose_matched_contrast_control import coefficients,validate,KEYS,SOURCE_KEYS,GLOBAL_KEYS,VERSION

class DoseTests(unittest.TestCase):
    def inputs(self):
        g=([.75,1.,1.25]*960);t=[v if i%7<3 else 1. for i,v in enumerate(g)]
        return g,t
    def test_exact_window_factor_counts_and_determinism(self):
        g,t=self.inputs();a=coefficients(7,g,t)
        self.assertEqual(a,coefficients(7,g,t));self.assertNotEqual(a,t)
        self.assertNotEqual(a,coefficients(17,g,t))
        for i in range(0,2880,300):self.assertEqual(Counter(a[i:i+300]),Counter(t[i:i+300]))
        self.assertTrue(all(x==1 or x==y for x,y in zip(a,g)))
    def test_bad_coefficients_and_length_rejected(self):
        g,t=self.inputs()
        for a,b in [(g[:-1],t),(g,t[:-1]),([2.]+g[1:],t),(g,[1.25]+t[1:])]:
            with self.assertRaises(ValueError):coefficients(7,a,b)
    def test_exposures_full_labels_configuration_unchanged(self):
        g,t=self.inputs();old={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')}
        old['pool_rows']=[{'member_id':'a','labels':[[1,.1,.2,.3,.4]]}];old['configuration']={'augmentation':'routed','lr':.00025}
        for f in ('schedules','exposures','windows','listings'):old[f]={k:['a']*2880 for k in SOURCE_KEYS}
        old['contrast']={k:t for k in SOURCE_KEYS};global_p=copy.deepcopy(old)
        for f in ('schedules','exposures','windows','listings'):global_p[f]={k:['a']*2880 for k in GLOBAL_KEYS}
        global_p['contrast']={k:g for k in GLOBAL_KEYS};p=copy.deepcopy(old)
        for f in ('schedules','exposures','windows','listings'):p[f]={k:['a']*2880 for k in KEYS}
        p['configuration']['augmentation']=VERSION;p['contrast']={k:coefficients(s,g,t) for s,k in zip((7,17,27),KEYS)}
        validate(old,global_p,p)
        for field in ('schedules','exposures','windows','listings','contrast'):
            bad=copy.deepcopy(p);bad[field][KEYS[0]]=[]
            with self.assertRaises(ValueError):validate(old,global_p,bad)
        bad=copy.deepcopy(p);bad['pool_rows'][0]['labels']=[]
        with self.assertRaises(ValueError):validate(old,global_p,bad)

if __name__=='__main__':unittest.main()
