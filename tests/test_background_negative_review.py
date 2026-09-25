import unittest


class BackgroundNegativeReviewTests(unittest.TestCase):
    def test_partition_is_14_accepted_1_held(self):
        held = {11}
        all_ids = {2,4,5,6,7,8,9,10,11,12,13,14,16,17,19}
        self.assertEqual(len(all_ids - held), 14)
        self.assertEqual(len(all_ids & held), 1)
