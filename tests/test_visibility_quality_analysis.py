import unittest
from scripts.vision.analyze_visibility_quality_results import paired_change


class PairedChangeTests(unittest.TestCase):
    def row(self, hit=False):
        return dict(pair_id='p1', variant='original', truth=[{'class_name':'reactor'}], image_sha256='abc', planned_assigned_hit=hit)

    def test_gain_loss(self):
        self.assertEqual(paired_change([self.row()], [self.row(True)]), {'original:gain':1})
        self.assertEqual(paired_change([self.row(True)], [self.row()]), {'original:loss':1})

    def test_missing_duplicate_and_stale(self):
        with self.assertRaises(ValueError): paired_change([self.row()], [])
        with self.assertRaises(ValueError): paired_change([self.row()]*2, [self.row()]*2)
        with self.assertRaises(ValueError): paired_change([self.row()], [{**self.row(), 'image_sha256':'changed'}])


if __name__ == '__main__': unittest.main()
