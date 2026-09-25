import unittest
from unittest.mock import patch
from scripts.vision.evaluate_hard_negative_coverage import validate


class EvaluationGateTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(view_id=str(i), variant='v', image_sha256=str(i), matching_conflict=False) for i in range(48)]
        r = dict(status='complete', cell='N', inputs={}, rows=rows, negative_rows=list(rows), matching_conflicts=0)
        return r, [(x, []) for x in rows], list(rows)

    @patch('scripts.vision.evaluate_hard_negative_coverage.verify')
    def test_complete_and_missing(self, verify):
        r, paired, neg = self.fixture()
        validate(r, 'N', paired, neg, {})
        r['rows'] = r['rows'][:-1]
        with self.assertRaises(ValueError): validate(r, 'N', paired, neg, {})

    @patch('scripts.vision.evaluate_hard_negative_coverage.verify')
    def test_duplicate_stale_and_conflict(self, verify):
        for mutation in ('duplicate', 'stale', 'conflict'):
            r, paired, neg = self.fixture()
            if mutation == 'duplicate': r['rows'] = r['rows'][:-1]+[r['rows'][0]]
            if mutation == 'stale': r['inputs'] = {'changed': 'digest'}
            if mutation == 'conflict': r['matching_conflicts'] = 1
            with self.assertRaises(ValueError): validate(r, 'N', paired, neg, {})

    def test_hash_failure_propagates(self):
        r, paired, neg = self.fixture()
        with patch('scripts.vision.evaluate_hard_negative_coverage.verify', side_effect=ValueError('hash')):
            with self.assertRaises(ValueError): validate(r, 'N', paired, neg, {})
