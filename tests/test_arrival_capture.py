import unittest
from src.flight.arrival_capture import ArrivalCapture


class ArrivalCaptureTests(unittest.TestCase):
    def test_enter_hold_and_reset(self):
        a=ArrivalCapture();self.assertFalse(a.update(.081));self.assertTrue(a.update(.08))
        self.assertTrue(a.update(.03));self.assertTrue(a.update(.031))
        self.assertTrue(a.update(.12))
        with self.assertRaises(RuntimeError):a.update(.12001)
        a.reset();self.assertFalse(a.update(.2))

    def test_invalid(self):
        for x in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):ArrivalCapture().update(x)
