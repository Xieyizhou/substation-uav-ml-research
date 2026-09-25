import unittest
from src.sensors.gazebo_lidar_observed import ArrivalTimes


class ArrivalObservationTests(unittest.TestCase):
    def test_observer_poll_gap_does_not_become_transport_gap(self):
        a=ArrivalTimes()
        for i in range(30):a.append(i*.033)
        self.assertAlmostEqual(a.maximum_gap_s,.033)
        self.assertEqual(a.long_gaps,[])
        a.append(2.)
        self.assertGreater(a.maximum_gap_s,.5)
        self.assertEqual(len(a.long_gaps),1)
        a.reset_measurement();a.append(4.);a.append(4.02)
        self.assertAlmostEqual(a.maximum_gap_s,.02)
