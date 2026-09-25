import unittest

from src.vision.canonical.expansion import balanced_select, target_candidates


class ExpansionTests(unittest.TestCase):
    def test_deterministic_and_avoids_blocker(self):
        objects = [{'name': 't', 'category': 'transformer', 'bounds': [4, 6, 4, 6, 0, 2]},
                   {'name': 'b', 'category': 'pole', 'bounds': [1, 2, 4, 6, 0, 4]}]
        one = target_candidates('m', objects, (0, 0), (12, 12), 1)
        self.assertEqual(one, target_candidates('m', objects, (0, 0), (12, 12), 1))
        self.assertTrue(one)

    def test_balanced_and_seen(self):
        rows = [{'category': c, 'view_id': c + str(i)} for c in ('transformer', 'switchgear') for i in range(3)]
        selected = balanced_select(rows, [], {'transformer0'}, 4)
        self.assertEqual(len(selected), 4)
        self.assertNotIn('transformer0', {r['view_id'] for r in selected})
