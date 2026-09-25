import unittest
from src.flight.tracking_envelope import check_route,segment_intersects_box


class TrackingEnvelopeTests(unittest.TestCase):
    def setUp(self):self.box=dict(name='obstacle',x0=3.,x1=4.,y0=3.,y1=4.)

    def test_centerline_pass_can_fail_tracking_tube(self):
        route=[(1.1,1.),(1.1,6.)]
        self.assertTrue(check_route(route,[self.box],tracking_error=0)['passed'])
        self.assertFalse(check_route(route,[self.box])['passed'])

    def test_contact_crossing_and_zero_length(self):
        self.assertTrue(segment_intersects_box((1.02,1),(1.02,6),self.box,1.98))
        self.assertTrue(segment_intersects_box((0,3.5),(7,3.5),self.box,0))
        self.assertTrue(segment_intersects_box((3,3),(3,3),self.box,0))
        self.assertFalse(segment_intersects_box((1,1),(1,1),self.box,0))
        self.assertFalse(segment_intersects_box((0,2.9),(7,2.9),self.box,0))

    def test_wide_route_passes_without_reducing_budget(self):
        self.assertTrue(check_route([(0.9,1),(0.9,6)],[self.box])['passed'])
        self.assertFalse(check_route([(0.6,1),(0.6,6)],[])['passed'])

    def test_uncertainty_cannot_make_route_safer(self):
        r=[(.9,1),(.9,6)]
        self.assertFalse(check_route(r,[self.box],map_uncertainty=.2)['passed'])
        for value in (-.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):check_route(r,[self.box],tracking_error=value)
