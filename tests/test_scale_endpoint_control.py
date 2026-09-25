import unittest
from unittest.mock import patch
from collections import Counter
from scripts.vision import scale_endpoint_control as c

class EndpointTests(unittest.TestCase):
    def test_positions_and_exact_dose(self):
        seq=[320,640,960]*150
        small,large=c.sizes(seq,'small'),c.sizes(seq,'large')
        self.assertEqual(Counter(small),{320:300,640:150})
        self.assertEqual(Counter(large),{960:300,640:150})
        self.assertEqual([i for i,x in enumerate(small) if x!=640],[i for i,x in enumerate(large) if x!=640])
        self.assertEqual(seq,[320,640,960]*150)
    def test_reject_invalid(self):
        for seq,arm in (([640]*450,'small'),([320,640,960]*150,'other'),([320,640,960]*149,'large')):
            with self.assertRaises(ValueError):c.sizes(seq,arm)
    def test_default_no_training(self):
        with patch('sys.argv',['entry']),patch.object(c,'freeze'),patch.object(c,'run') as run,patch.object(c,'train') as train:
            c.main()
        run.assert_not_called();train.assert_not_called()
    def test_no_preflight_no_training_completion(self):
        with patch.object(c,'verify_check',side_effect=ValueError('stale')):
            with self.assertRaises(ValueError):c.complete('small-7')

if __name__=='__main__':unittest.main()
