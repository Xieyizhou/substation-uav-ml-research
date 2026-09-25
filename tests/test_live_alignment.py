import math
import unittest
from scripts.flight.check_live_alignment import evaluate


class LiveAlignmentTests(unittest.TestCase):
    def samples(self):
        a=[dict(timestamp=1000000+i*50000,x=0.,y=0.,heading=math.pi/2,xy_reset_counter=0,heading_reset_counter=0) for i in range(12)]
        b=[dict(r,x=-8.5,y=-8.5,timestamp=r['timestamp']+1000) for r in a]
        return a,b

    def test_pass(self):
        self.assertTrue(evaluate(*self.samples())['passed'])

    def test_heading_and_reset_reject(self):
        a,b=self.samples();a[-1]['heading']+=.1
        self.assertFalse(evaluate(a,b)['passed'])
        a,b=self.samples();a[-1]['xy_reset_counter']=1
        self.assertFalse(evaluate(a,b)['passed'])

    def test_missing_nonfinite_and_no_sync_reject(self):
        a,b=self.samples();a[0].pop('xy_reset_counter')
        self.assertFalse(evaluate(a,b)['passed'])
        a,b=self.samples();a[0]['heading']=math.nan
        with self.assertRaises(ValueError):evaluate(a,b)
        a,b=self.samples()
        with self.assertRaises(ValueError):evaluate(a,b[:2])
