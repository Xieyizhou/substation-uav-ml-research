import unittest
from scripts.vision.freeze_annotation_revision_design import inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.pool = [dict(member_id='a', lineage_id='pose:a'),
                     dict(member_id='b', lineage_id='pose:a'),
                     dict(member_id='c', lineage_id='c')]
        self.frames = [dict(review_ids=['T027'], member_id='a', lineage_id='pose:a'),
                       dict(review_ids=['T036'], member_id='c', lineage_id='pose:c')]

    def test_bounded_lineage_and_planned_not_actual(self):
        rows = inventory(self.pool, self.frames, {'seed7': ['a', 'a', 'c']})
        self.assertEqual([len(r['members']) for r in rows], [2, 1])
        self.assertEqual(rows[0]['members'][1]['planned_exposures']['seed7'], 0)
        self.assertFalse(rows[0]['actual_exposures_verified'])

    def test_duplicate_member_rejected(self):
        with self.assertRaises(ValueError):
            inventory(self.pool + self.pool[:1], self.frames, {})

    def test_unknown_draw_rejected(self):
        with self.assertRaises(ValueError):
            inventory(self.pool, self.frames, {'seed7': ['missing']})

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            inventory(self.pool[:-1], self.frames, {})

    def test_duplicate_target_rejected(self):
        with self.assertRaises(ValueError):
            inventory(self.pool, self.frames + self.frames[:1], {})
