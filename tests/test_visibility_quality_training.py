import unittest
from collections import Counter
from scripts.vision.run_visibility_quality_training import filtered_draws

class QualityScheduleTests(unittest.TestCase):
    def test_exact_quota_exclusion_and_prefix(self):
        rows=[dict(member_id=f'{s}{i}',subset=s) for s in ('base','regular','bridge_positive','hard_negative') for i in range(3)]
        old=[r['member_id'] for r in rows]*50
        excluded={'base0','regular1'};result,changes=filtered_draws(rows,old,excluded,7)
        self.assertEqual(len(result),600);self.assertFalse(set(result)&excluded)
        lookup={r['member_id']:r['subset'] for r in rows}
        self.assertEqual(Counter(lookup[x] for x in result),Counter(lookup[x] for x in old))
        self.assertEqual(len(changes),100);self.assertEqual((result*3)[:600],result)
        self.assertEqual(filtered_draws(rows,old,excluded,7)[0],result)

    def test_empty_subset_fails(self):
        rows=[dict(member_id='base0',subset='base')]
        with self.assertRaises(ValueError):filtered_draws(rows,['base0']*600,{'base0'},7)
