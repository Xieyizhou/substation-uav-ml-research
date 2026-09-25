import unittest
from unittest.mock import patch
from scripts.vision import evaluate_frozen_multiscale as e

class EvaluationTests(unittest.TestCase):
    def test_default_never_infers(self):
        with patch('sys.argv',['evaluate']), patch.object(e,'complete') as c, patch.object(e,'evaluate') as run:
            e.main()
        self.assertEqual(c.call_count,6)
        run.assert_not_called()

    def test_training_forbidden(self):
        with self.assertRaises(AssertionError): e.forbidden()

    def test_incomplete_and_duplicate_rejected(self):
        r=dict(status='complete',cell='fixed-7',rows=[],negative_rows=[])
        with patch.object(e.prior,'verify'):
            with self.assertRaises(ValueError): e.validate_record(r,'fixed-7')
            r.update(rows=[dict(pair_id='one',variant='original')]*48,
                     negative_rows=[dict(view_id='one',variant='original')]*48)
            with self.assertRaises(ValueError): e.validate_record(r,'fixed-7')

    def test_stale_identity_rejected(self):
        with patch.object(e.prior,'verify',side_effect=ValueError('stale')):
            with self.assertRaises(ValueError): e.validate_record({},'fixed-7')

if __name__=='__main__': unittest.main()
