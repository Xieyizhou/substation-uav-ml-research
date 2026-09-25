import unittest
from scripts.vision.finalize_depth_clip_test import compare,OUT,controls,base
from scripts.vision.run_depth_clip_clock_fence import valid_clock


class DepthClipTests(unittest.TestCase):
    def records(self,tag='T027'):
        n=2 if tag=='T020' else 1
        return [base.read(controls(tag)),base.read(OUT/'replay'/tag/f'attempt-{n:02}/receipt.json'),base.read(controls(tag,'clipped'))]

    def test_four_real_poses(self):
        for tag in ('T020','T036','T023','T027'):
            result=compare(tag,*self.records(tag));self.assertTrue(result['technical_gate_passed'])
            self.assertEqual(result['comparisons'][0]['max_depth_delta'],0)

    def test_missing_128_held(self):
        a,b,c=self.records();del b['records'][0]['full_boxes']['128']
        self.assertFalse(compare('T027',a,b,c)['technical_gate_passed'])

    def test_existing_drift_held(self):
        a,b,c=self.records();b['records'][0]['full_boxes']['113'][0]+=2
        self.assertFalse(compare('T027',a,b,c)['technical_gate_passed'])

    def test_incomplete_rejected(self):
        a,b,c=self.records();b['records'].pop()
        with self.assertRaises(ValueError):compare('T027',a,b,c)

    def test_bad_sync_rejected(self):
        a,b,c=self.records();b['records'][0]['skew_ms']=40
        with self.assertRaises(ValueError):compare('T027',a,b,c)

    def test_clock_missing_not_accepted(self):
        self.assertFalse(valid_clock({'header':{'stamp':{'nsec':100}}}))
        self.assertFalse(valid_clock({}))
        self.assertFalse(valid_clock({'header':{'stamp':{'sec':0,'nsec':100}}}))

    def test_valid_explicit_clock(self):
        self.assertTrue(valid_clock({'header':{'stamp':{'sec':'1','nsec':100}}}))

    def test_nonfinite_invalid_clock(self):
        for stamp in ({'sec':float('nan')},{'sec':1,'nsec':-1},{'sec':1,'nsec':1e9}):
            self.assertFalse(valid_clock({'header':{'stamp':stamp}}))


if __name__=='__main__':unittest.main()
