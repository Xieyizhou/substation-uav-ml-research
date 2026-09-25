import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from scripts.flight.prepare_lowload_vehicle import verify_change,MODEL

class LowloadVehicleTests(unittest.TestCase):
    def setUp(self):
        self.original=ET.parse(Path('simulation/models/x500_research/model.sdf'))
        self.changed=copy.deepcopy(self.original)
        self.changed.getroot().find('model').set('name',MODEL)
        sensor=self.changed.getroot().find("model/link[@name='research_camera_link']/sensor[@name='research_rgb']")
        sensor.find('camera/image/width').text='640';sensor.find('camera/image/height').text='360'

    def test_only_rgb_resolution_allowed(self):
        verify_change(self.original,self.changed)

    def test_other_field_and_bad_resolution_rejected(self):
        self.changed.getroot().find('model').set('unknown','changed')
        with self.assertRaisesRegex(ValueError,'Non-allowed'):verify_change(self.original,self.changed)
        del self.changed.getroot().find('model').attrib['unknown']
        self.changed.getroot().find("model/link[@name='research_camera_link']/sensor[@name='research_rgb']/camera/image/width").text='320'
        with self.assertRaisesRegex(ValueError,'resolution'):verify_change(self.original,self.changed)
