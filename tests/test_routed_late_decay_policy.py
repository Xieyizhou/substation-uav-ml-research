import unittest
from scripts.vision.routed_late_decay_policy import BASE_LR,factor,sequence

class LateDecayTest(unittest.TestCase):
    def test_exact_budget_prefix_and_endpoint(self):
        values=sequence()
        self.assertEqual(len(values),480)
        self.assertEqual(values[:240],[BASE_LR]*240)
        self.assertEqual(values[-10:],[BASE_LR/2]*10)
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        for epoch in range(48):
            self.assertEqual(values[epoch*10:(epoch+1)*10],[BASE_LR*factor(epoch)]*10)

    def test_out_of_protocol_fails(self):
        for value in (-1,48,1.5,True,'1'):
            with self.assertRaises(ValueError):factor(value)
