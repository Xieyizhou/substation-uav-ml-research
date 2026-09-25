import unittest
from src.flight.turn_hold import slew_heading


class TurnHoldTests(unittest.TestCase):
    def test_rate_and_long_pause_bounded(self):
        self.assertEqual(slew_heading(90, -90, .05), 89)
        self.assertEqual(slew_heading(90, -90, 10), 88)

    def test_wrap_and_target(self):
        self.assertEqual(slew_heading(179, -179, .1), -179)
        self.assertAlmostEqual(slew_heading(10, 10.2, .05), 10.2)

    def test_invalid(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                slew_heading(value, 0, .05)
        with self.assertRaises(ValueError):
            slew_heading(0, 0, -1)
