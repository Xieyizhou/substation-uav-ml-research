from tests.test_live_alignment import LiveAlignmentTests
from scripts.flight.check_final_heading import assess
from scripts.flight.fly_px4_shadow_hover import guard
import unittest


class FinalHeadingTests(unittest.TestCase):
    def test_low_error_cannot_replace_final_alignment(self):
        a,b=LiveAlignmentTests().samples()
        self.assertFalse(assess(a,b)['horizontal_flight_ready'])
        for r in a:r['heading_good_for_control']=True
        self.assertTrue(assess(a,b)['horizontal_flight_ready'])
        a[-1]['heading_reset_counter']=1
        self.assertFalse(assess(a,b)['horizontal_flight_ready'])

    def test_two_meter_envelope_stays_bounded(self):
        origin=dict(north=0.,east=0.,down=0.)
        state=dict(local=(10.,dict(north=0.,east=0.,down=-2.)),attitude=(10.,dict(roll=0.,pitch=0.)))
        with self.assertRaises(RuntimeError):guard(state,10.,origin)
        self.assertEqual(guard(state,10.,origin,max_height_m=2.8),(2.,0.))
        state['local'][1]['down']=-2.81
        with self.assertRaises(RuntimeError):guard(state,10.,origin,max_height_m=2.8)
