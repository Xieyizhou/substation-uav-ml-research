import unittest


class MemorizationReviewTests(unittest.TestCase):
    def test_unmatched_partition_is_exhaustive(self):
        wrong = {1, 3, 15, 18}
        background = set(range(1, 20)) - wrong
        self.assertEqual(len(wrong), 4)
        self.assertEqual(len(background), 15)
        self.assertFalse(wrong & background)
