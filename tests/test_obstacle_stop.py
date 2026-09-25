import math
from types import SimpleNamespace
import unittest
from scripts.flight.fly_obstacle_stop import nearest


class ObstacleStopTests(unittest.TestCase):
    def source(self,values,age=.1,healthy=True):
        scan=SimpleNamespace(ranges_m=values,range_min_m=.1,range_max_m=30.,age_s=lambda now:age)
        return SimpleNamespace(latest=lambda:scan,health=lambda:SimpleNamespace(healthy=healthy))

    def test_retains_dangerous_range_for_stop_decision(self):
        self.assertEqual(nearest(self.source([1.4]*100)),1.4)
        self.assertEqual(nearest(self.source([math.inf]*99+[1.5])),1.5)

    def test_rejects_invalid_and_stale(self):
        for values in ([0.]*100,[-math.inf]*100,[math.inf]*100,[math.nan]*100,[1.]):
            with self.assertRaises(RuntimeError):nearest(self.source(values))
        with self.assertRaises(RuntimeError):nearest(self.source([2.]*100,age=.51))
        with self.assertRaises(RuntimeError):nearest(self.source([2.]*100,healthy=False))
