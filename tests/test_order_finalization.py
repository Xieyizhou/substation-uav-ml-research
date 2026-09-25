import unittest
from scripts.vision.finalize_order_retention import reconstruct
from scripts.vision.exposure_order_retention import OUT,read,permutation

class FinalizationTests(unittest.TestCase):
    def test_flatten_uses_selected_batch(self):
        self.assertEqual(reconstruct([['a','b'],['c','d']],[1,0]),['c','d','a','b'])
    def test_frozen_real_permutations(self):
        p=read(OUT/'protocol.json')
        for seed in (7,17,27):
            a=p['schedules'][f'staged-450-{seed}'];b,ids=permutation(a,p['variant_by_member'],seed)
            self.assertEqual(p['schedules'][f'interleaved-450-{seed}'],reconstruct(b,ids))

if __name__=='__main__':unittest.main()
