import unittest
from scripts.vision.small_scale_schedule import tails, validate_tail


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(member_id=f'{p}-{l}-{v}', pair_id=str(p), illumination=l,
            variant=v, lineage_id=str(p), class_instances={'reactor': 1}, label_sha256=str(p))
            for p in range(4) for l in ('normal', 'cool') for v in ('original', 'gray035')]

    def test_exact_reproducible_all_seeds(self):
        for seed in (7, 17, 27):
            a, b = tails(self.rows, seed)
            self.assertEqual((a, b), tails(self.rows, seed))
            validate_tail(a, b, self.rows)

    def test_missing_duplicate(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError): tails(rows, 7)

    def test_label_drift(self):
        self.rows[1]['label_sha256'] = 'changed'
        with self.assertRaises(ValueError): tails(self.rows, 7)

    def test_position_drift(self):
        a, b = tails(self.rows, 7)
        b[0] = next(r['member_id'] for r in self.rows if r['pair_id'] != a[0].split('-')[0])
        with self.assertRaises(ValueError): validate_tail(a, b, self.rows)


if __name__ == '__main__': unittest.main()
