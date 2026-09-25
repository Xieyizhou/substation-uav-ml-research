import unittest
from unittest.mock import patch
from scripts.vision import train_interleaved_small_scale as t


class Entry(unittest.TestCase):
    def test_unknown(self):
        with self.assertRaises(ValueError): t.complete('SR1100-7')

    def test_readiness_missing_draws(self):
        r=dict(status='ready_for_training_not_started',cells=list(t.KEYS),actual_draws_verified=6600)
        with patch.object(t.prior,'read',return_value=r),patch.object(t.prior,'verify'):
            with self.assertRaises(ValueError): t.ready()

    def test_actual_mismatch(self):
        c=dict(exposure_path='unused',optimizer_steps=1100,weights='w',weights_sha256='sha')
        x=dict(actual=['wrong'],brightness_log=[],optimizer_steps=1100)
        with patch.object(t,'freeze',return_value={}),patch.object(t.prior,'read',side_effect=[c,x]),patch.object(t.prior,'verify'),patch.object(t,'check',side_effect=ValueError('actual')):
            with self.assertRaisesRegex(ValueError,'actual'): t.complete('ISR1100-7')

    def test_wrong_endpoint(self):
        c=dict(exposure_path='unused',optimizer_steps=1000)
        x=dict(actual=[],brightness_log=[],optimizer_steps=1000)
        with patch.object(t,'freeze',return_value={}),patch.object(t.prior,'read',side_effect=[c,x]),patch.object(t.prior,'verify'),patch.object(t,'check'):
            with self.assertRaisesRegex(ValueError,'endpoint'): t.complete('ISM1100-27')


if __name__ == '__main__': unittest.main()
