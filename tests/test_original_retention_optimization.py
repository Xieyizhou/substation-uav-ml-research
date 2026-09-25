import unittest
from scripts.vision.design_original_retention_optimization import build


class RetentionDesignTests(unittest.TestCase):
    def test_prefix_common_slots_and_budget(self):
        rows = {'c': dict(subset='base', class_instances={'x': 1}),
                **{k: dict(subset='bridge_positive', lineage_id='g', class_instances={'x': 2})
                   for k in ('o', 'n', 'b')}}
        original = ['c', 'c', 'c', 'c', 'o'] * 360
        a, b = build(original, rows, {('g', 'neutral_bridge'): 'n', ('g', 'background_bridge'): 'b'})
        self.assertEqual(len(a), 2700)
        self.assertEqual(b[:1800], original)
        self.assertEqual(sum(x != y for x, y in zip(a, b)), 180)
        self.assertEqual(b.count('n'), 90)
        self.assertEqual(b.count('b'), 90)
        self.assertTrue(all(x == y for x, y in zip(a, b) if x == 'c'))

    def test_incomplete_prefix_rejected(self):
        with self.assertRaises(ValueError):
            build([], {}, {})
