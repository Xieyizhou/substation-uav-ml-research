import math
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from scripts.flight.check_vehicle_envelope import transform,analyze


class VehicleEnvelopeTests(unittest.TestCase):
    def xml(self):
        m=ET.Element('model',name='test');r=ET.Element('sdf');r.append(m)
        for name in ('base_link','research_lidar_link','research_camera_link',*[f'rotor_{i}' for i in range(4)]):
            link=ET.SubElement(m,'link',name=name)
            ET.SubElement(link,'pose').text='0 0 .24 0 0 0'
            count=5 if name=='base_link' else (1 if name.startswith('rotor') else 0)
            for i in range(count):
                c=ET.SubElement(link,'collision',name=str(i));g=ET.SubElement(c,'geometry');b=ET.SubElement(g,'box');ET.SubElement(b,'size').text='.2 .2 .2'
            if name.startswith('rotor'):
                j=ET.SubElement(m,'joint',name=name+'_joint',type='revolute')
                ET.SubElement(j,'parent').text='base_link';ET.SubElement(j,'child').text=name
        return r

    def test_fixed_vertices_and_rotor_sweep_are_bounded(self):
        a=analyze(ET.tostring(self.xml(),encoding='unicode'))
        self.assertAlmostEqual(a['collision_sphere_radius_m'],.24+math.sqrt(3)*.1)
        self.assertEqual(sum(b['rotating'] for b in a['collision_bounds']),4)
        self.assertAlmostEqual(a['collision_bounds'][0]['origin_sphere_radius_m'],math.sqrt(.34**2+.1**2+.1**2))

    def test_unknown_frames_and_missing_shapes_fail(self):
        r=self.xml();r.find('model/link/pose').set('relative_to','missing')
        with self.assertRaises(ValueError):analyze(ET.tostring(r,encoding='unicode'))
        r=self.xml();link=r.find('model/link');link.remove(link.find('collision'))
        with self.assertRaises(ValueError):analyze(ET.tostring(r,encoding='unicode'))

    def test_pose_rotation_and_nonfinite(self):
        p=ET.fromstring('<pose>1 2 3 0 0 1.5707963267948966</pose>')
        np.testing.assert_allclose(transform(p)@np.array([1,0,0,1]),[1,3,3,1],atol=1e-10)
        with self.assertRaises(ValueError):transform(ET.fromstring('<pose>nan 0 0 0 0 0</pose>'))
