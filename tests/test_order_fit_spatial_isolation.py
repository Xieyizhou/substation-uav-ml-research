import unittest
from scripts.vision.isolate_order_fit_spatial_risks import isolate


class SpatialIsolationTests(unittest.TestCase):
    def setUp(self):
        self.members = [dict(member_id=x, lineage_id=g, lineage_resolution='member_only_no_independence_claim')
                        for x, g in [('a', 'g'), ('b', 'g'), ('c', 'h')]]
        self.old = [dict(member_id=x, status='quarantined' if x == 'c' else 'pending', reasons=[])
                    for x in ['a', 'b', 'c']]
        self.decision = dict(member_id='a', object_id='transformer', status='hold_pending_named_target_coverage_risk')

    def test_propagates_and_preserves_old_holds(self):
        rows, unresolved = isolate(self.members, self.old, [self.decision])
        self.assertTrue(all(r['status'] == 'quarantined' and not r['training_eligible'] for r in rows))
        self.assertEqual(unresolved, ['a'])

    def test_background_is_not_admission(self):
        rows, _ = isolate(self.members, self.old, [dict(self.decision, status='named_structure_is_source_defined_non_target_cabinet')])
        self.assertEqual(rows[0]['status'], 'pending')
        self.assertFalse(rows[0]['training_eligible'])

    def test_duplicate_and_missing_rejected(self):
        with self.assertRaises(ValueError): isolate(self.members, self.old, [self.decision, self.decision])
        with self.assertRaises(ValueError): isolate(self.members, self.old[:-1], [self.decision])


if __name__ == '__main__': unittest.main()
