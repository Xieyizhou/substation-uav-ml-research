import unittest
from unittest.mock import patch
from scripts.vision import endpoint_member_fit as fit

class EndpointFitTests(unittest.TestCase):
    def test_default_does_not_infer(self):
        with patch('sys.argv',['entry']),patch.object(fit,'freeze',return_value={}),patch.object(fit,'check'),patch.object(fit,'infer') as infer,patch.object(fit,'finish') as finish:
            fit.main()
        infer.assert_not_called();finish.assert_not_called()
    def test_all_nine_weights_preserved(self):
        self.assertEqual(set(fit.KEYS),{f'{a}-{s}' for a in ('fixed','small','large') for s in (7,17,27)})
    def test_labels_rechecked(self):
        with patch.object(fit.prior,'verify'),patch.object(fit,'reviewed_input'),patch.object(fit,'truth_for',return_value=['changed']):
            with self.assertRaises(ValueError):fit.check({'members':[{'truth':[]}]})

if __name__=='__main__':unittest.main()
