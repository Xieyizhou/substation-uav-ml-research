import unittest
from collections import Counter
from scripts.vision.run_fixed_budget_diagnosis import repeat_sequence, retention, NAMES


class FixedBudgetTests(unittest.TestCase):
    def test_exact_blocks_prefix_and_quota(self):
        source = [str(i % 47) for i in range(600)]
        actual = repeat_sequence(source)
        self.assertEqual(len(actual), 1800)
        for start in (0, 600, 1200):
            self.assertEqual(actual[start:start + 600], source)
        self.assertEqual(Counter(actual), Counter({k: v * 3 for k, v in Counter(source).items()}))

    def test_invalid_sequence(self):
        for source in ([], ['x'] * 599, ['x'] * 601, [''] * 600):
            with self.assertRaises(ValueError):
                repeat_sequence(source)

    def test_retention_boundary(self):
        def data(value):
            return {v: dict(instance_recall=value, per_class={n: dict(instance_recall=value) for n in NAMES}) for v in ('original', 'lighting')}
        self.assertTrue(all(c['passed'] for c in retention(data(.75), data(.8))))
        self.assertFalse(any(c['passed'] for c in retention(data(.749), data(.8))))


if __name__ == '__main__':
    unittest.main()
