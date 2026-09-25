import math
from types import SimpleNamespace
import unittest
from scripts.flight.fly_short_route import velocity,lidar_clearance


class ShortRouteTests(unittest.TestCase):
    def source(self,ranges,healthy=True,age=.1):
        scan=SimpleNamespace(ranges_m=ranges,range_min_m=.1,range_max_m=30.,age_s=lambda now:age)
        return SimpleNamespace(latest=lambda:scan,health=lambda:SimpleNamespace(healthy=healthy))

    def test_velocity_bounded(self):
        for n,e in ((0,0),(1,1),(100,-100)):
            a,b=velocity(n,e);self.assertLessEqual(math.hypot(a,b),.30000001)

    def test_sensor_gates(self):
        self.assertEqual(lidar_clearance(self.source([4.]*100)),4.)
        for source in (self.source([1.]*100),self.source([4.]*100,False),self.source([4.]*100,age=1),self.source([math.inf]*100),self.source([math.nan]*100),self.source([4.])):
            with self.assertRaises(RuntimeError):lidar_clearance(source)
