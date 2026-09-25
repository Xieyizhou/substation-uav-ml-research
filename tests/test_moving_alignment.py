import math
import unittest
from scripts.flight.diagnose_moving_alignment import interpolate

class MovingAlignmentTests(unittest.TestCase):
    def row(self,t,h):return dict(timestamp=t,heading=math.radians(h),x=t/10000,y=0.,z=-2.)
    def test_motion_time_offset_is_not_heading_bias(self):
        result=interpolate([self.row(0,0),self.row(20000,4)],10000)
        self.assertAlmostEqual(math.degrees(result['heading']),2.)
        self.assertAlmostEqual(result['x'],1.)
    def test_wrap_uses_short_arc(self):
        result=interpolate([self.row(0,179),self.row(20000,-179)],10000)
        self.assertAlmostEqual(abs(math.degrees(result['heading'])),180.)
    def test_gaps_duplicates_and_extrapolation_rejected(self):
        for rows,t in (([self.row(0,0),self.row(50000,4)],20000),([self.row(0,0),self.row(0,4)],0),([self.row(0,0),self.row(20000,4)],21000)):
            with self.assertRaises(ValueError):interpolate(rows,t)
