import unittest
from unittest.mock import patch
from scripts.vision import evaluate_small_scale as evaluation
from scripts.vision import train_small_scale as training


class EntryTest(unittest.TestCase):
    def test_old_endpoint_rejected(self):
        with patch.object(evaluation, 'protocol', return_value={}), \
             patch.object(evaluation.prior, 'verify'), \
             patch.object(evaluation.prior, 'read', side_effect=[
                 {'exposure_path':'unused', 'optimizer_steps':1000}, {'optimizer_steps':1000}]):
            with self.assertRaisesRegex(ValueError, 'Wrong endpoint'):
                evaluation.complete('SR1100-7')

    def test_unknown_unit_rejected(self):
        with self.assertRaises(ValueError): evaluation.complete('ICG1000-7')

    def test_incomplete_readiness_rejected(self):
        record = dict(status='ready_for_training_not_started', cells=list(training.KEYS), actual_draws_verified=18000)
        with patch.object(training.prior,'read',return_value=record), patch.object(training.prior,'verify'):
            with self.assertRaisesRegex(ValueError,'Readiness incomplete'): training.ready()

    def test_complete_endpoint_requires_actual_check(self):
        receipt = dict(optimizer_steps=1100,exposure_path='unused',weights='unused',weights_sha256='w')
        exposure = dict(optimizer_steps=1100,actual=['wrong'],brightness_log=[])
        with patch.object(evaluation,'protocol',return_value={}), patch.object(evaluation.prior,'verify'), \
             patch.object(evaluation.prior,'read',side_effect=[receipt,exposure]), \
             patch.object(evaluation,'check',side_effect=ValueError('Actual mismatch')):
            with self.assertRaisesRegex(ValueError,'Actual mismatch'): evaluation.complete('SM1100-7')


if __name__ == '__main__': unittest.main()
