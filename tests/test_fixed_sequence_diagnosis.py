import unittest
from collections import Counter
from scripts.vision.run_fixed_sequence_diagnosis import schedule,SOURCE,read,variation

class FixedSequenceTests(unittest.TestCase):
    def test_fresh_deterministic_exact_draws(self):
        prior=read(SOURCE/'protocol.json');rows=prior['pool_rows'];lookup={r['member_id']:r for r in rows}
        a,pairs=schedule(rows);b,_=schedule(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(len(a),600);self.assertEqual(len(pairs),18)
        c=Counter('new' if m.startswith('coverage:') else 'old' if lookup[m]['subset']=='hard_negative' else lookup[m]['subset'] for m in a)
        self.assertEqual(c,dict(base=216,regular=156,bridge_positive=120,old=72,new=36))
        old=Counter(m for m in a if lookup[m]['subset']=='hard_negative' and not m.startswith('coverage:'))
        self.assertEqual(len(old),24);self.assertEqual(set(old.values()),{3})
    def test_variation_boundary(self):
        g={'no_target':{'frame_false_positive_rate':dict(max_seed=.125,min_seed=.125)}}
        p=dict(fpr_range_threshold=2/48,planned_range_threshold=2/12,recall_range_threshold=.05,variants=[])
        self.assertFalse(variation(g,p)['material_seed_variation'])
        g['no_target']['frame_false_positive_rate']['max_seed']=.125+2/48
        self.assertTrue(variation(g,p)['material_seed_variation'])
