import unittest
from collections import Counter
from scripts.vision.analyze_material_transfer_controls import quota_triads,dimensions
from scripts.vision.finalize_material_transfer_controls import scale_order
from scripts.vision.run_material_scale_control import validate
from src.ml.artifacts import object_sha256

class TransferTests(unittest.TestCase):
    def test_budget(self):
        self.assertFalse(quota_triads(1)['feasible'])
        for n in (3,60,61,68):
            r=quota_triads(n);self.assertEqual(r['original']+r['warm']+r['cool'],n);self.assertGreater(min(r[k] for k in ('original','warm','cool')),0)

    def test_dimensions(self):
        r=dimensions([0,0,300,150],1920,1080)
        self.assertEqual(r['short640'],50);self.assertEqual(r['aspect'],2)

    def test_exact_reproducible_scales(self):
        for s in (7,17,27):
            r=scale_order(s);self.assertEqual(Counter(r),{320:150,640:150,960:150});self.assertEqual(r,scale_order(s))
        self.assertNotEqual(scale_order(7),scale_order(17))

    def test_missing_scale_rejected(self):
        p=dict(identity='p',scales=[320,960],rows=[dict(id='x')]);r=dict(protocol_identity='p',model='m',status='complete',rows=[],inputs={})
        r['identity']=object_sha256(r)
        with self.assertRaises(ValueError):validate(r,p,'m')

if __name__=='__main__':unittest.main()
