"""Integration regression against frozen local research members."""
import unittest
from collections import Counter
from pathlib import Path
from scripts.vision.design_original_retention_optimization_v2 import OUT
from scripts.vision.run_visibility_repair_training_v2 import OUT as SOURCE, read

class RealQuotaTests(unittest.TestCase):
    def test_all_seeds_exact_quota_and_labels(self):
        d = read(OUT/'design.json'); p = read(SOURCE/'protocol.json')
        rows = {r['member_id']:r for r in p['pool_rows']}
        for seed in (7,17,27):
            a=d['schedules'][f'retained_reference-450-{seed}']
            b=d['schedules'][f'retained_appearance-450-{seed}']
            self.assertEqual(len(a),2700); self.assertEqual(len(b),2700)
            self.assertEqual(a[:1800],p['schedules'][f'original_only-300-{seed}'])
            self.assertEqual(b[:1800],a[:1800])
            self.assertEqual(Counter(rows[m]['subset'] for m in a[1800:]),
                Counter(base=324,regular=234,bridge_positive=180,hard_negative=162))
            self.assertEqual(sum(x!=y for x,y in zip(a,b)),180)
            for x,y in zip(a,b):
                if x==y: continue
                self.assertEqual(rows[x]['lineage_id'],rows[y]['lineage_id'])
                self.assertEqual(Path(rows[x]['label_path']).read_bytes(),Path(rows[y]['label_path']).read_bytes())

if __name__=='__main__':unittest.main()
