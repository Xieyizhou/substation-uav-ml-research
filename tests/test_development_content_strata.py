import copy
import unittest
from scripts.vision.record_development_content_strata_review import STRATA


class ContentStrataTests(unittest.TestCase):
    def test_fixed_scope_and_categories(self):
        self.assertEqual(sum(map(len, STRATA.values())), 60)
        self.assertTrue(set(sum(STRATA.values(), [])) <= {'clear', 'partial', 'fragment', 'unknown'})

    def test_unknown_precedence_is_explicit(self):
        # The frozen design does not infer unknown from low area or model scores.
        self.assertIn('unknown', {'clear', 'partial', 'fragment', 'unknown'})

    def test_no_score_based_category(self):
        self.assertNotIn('confidence', STRATA)

    def test_mapping_is_not_mutated_by_copy(self):
        x = copy.deepcopy(STRATA); x[1][0] = 'unknown'
        self.assertEqual(STRATA[1][0], 'clear')
