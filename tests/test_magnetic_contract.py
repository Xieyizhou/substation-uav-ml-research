from pathlib import Path
import tempfile
import unittest
from scripts.flight.retest_magnetic_contract import validate_contract
from scripts.flight.probe_magnetic_contract import decode


class MagneticContractTests(unittest.TestCase):
    def test_requires_explicit_matching_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'server.config'
            body='<plugin name="gz::sim::systems::Magnetometer"><use_earth_frame_ned>false</use_earth_frame_ned><use_units_gauss>true</use_units_gauss></plugin>'
            p.write_text('<server_config><plugins>'+body+'</plugins></server_config>')
            validate_contract(p)
            p.write_text('<server_config><plugins>'+body+body+'</plugins></server_config>')
            with self.assertRaises(ValueError):validate_contract(p)
            p.write_text('<server_config><plugins>'+body.replace('false','true')+'</plugins></server_config>')
            with self.assertRaises(ValueError):validate_contract(p)

    def test_invalid_sensor_data(self):
        self.assertEqual(decode({'fieldTesla':{'x':1,'y':2,'z':3}}),[1.,2.,3.])
        with self.assertRaises(ValueError):decode({})
        with self.assertRaises(ValueError):decode({'fieldTesla':{'x':float('nan')}})
