import math
import unittest
from unittest.mock import patch
from scripts.flight.retest_sim_heading import measured_declination


class SimHeadingProfileTests(unittest.TestCase):
    def test_body_to_truth_declination(self):
        d=math.radians(-2.5);rows=[]
        for i in range(30):
            t=1000000+i*100000
            rows.append(('vehicle_attitude_groundtruth',0,dict(timestamp=t,q=(math.sqrt(.5),0,0,math.sqrt(.5)))))
            rows.append(('vehicle_magnetometer',0,dict(timestamp=t,magnetometer_ga=(math.sin(d),-math.cos(d),.4))))
        with patch('scripts.flight.retest_sim_heading.read_samples',return_value=iter(rows)):
            angle,count=measured_declination('unused')
        self.assertAlmostEqual(angle,-2.5);self.assertEqual(count,30)

    def test_insufficient_data_rejected(self):
        rows=[('vehicle_attitude_groundtruth',0,dict(timestamp=1000000,q=(1,0,0,0)))]
        with patch('scripts.flight.retest_sim_heading.read_samples',return_value=iter(rows)):
            with self.assertRaises(ValueError):measured_declination('unused')
