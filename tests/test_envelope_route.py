import json
import unittest
from pathlib import Path
from src.flight.envelope_route import envelope_route
from src.flight.tracking_envelope import check_route
from scripts.flight.measure_mapping_error import interpolate


class EnvelopeRouteTests(unittest.TestCase):
    def test_groundtruth_interpolation_rejects_unknown_gap(self):
        rows=[dict(timestamp=0,x=0,y=0,heading=0),dict(timestamp=20000,x=2,y=4,heading=0)]
        self.assertEqual(interpolate(rows,[0,20000],10000)['x'],1)
        self.assertIsNone(interpolate(rows,[0,20000],30000))
        rows[1]['timestamp']=50000
        self.assertIsNone(interpolate(rows,[0,50000],10000))
