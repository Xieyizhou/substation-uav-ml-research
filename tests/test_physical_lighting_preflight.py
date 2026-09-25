import unittest
import xml.etree.ElementTree as ET
from scripts.vision.prepare_physical_lighting_control import change_lighting,validate_change,slots

class LightingPreflightTests(unittest.TestCase):
    def world(self):
        return ET.fromstring('<sdf><world><scene><ambient>0.72 0.72 0.72 1</ambient><background>0.72 0.75 0.78 1</background></scene><light name="sun"><diffuse>0.85 0.85 0.82 1</diffuse><direction>1 0 -1</direction></light><model name="equipment"><pose>1 2 3 0 0 0</pose></model></world></sdf>')
    def test_only_frozen_lighting_changes(self):
        a=self.world();before=ET.tostring(a);b=change_lighting(a);validate_change(a,b)
        self.assertEqual(before,ET.tostring(a))
        b.find('.//model/pose').text='2 2 3 0 0 0'
        with self.assertRaises(ValueError):validate_change(a,b)
    def test_unfrozen_intensity_rejected(self):
        a=self.world();b=change_lighting(a);b.find('.//scene/ambient').text='0.1 0.1 0.1 1'
        with self.assertRaises(ValueError):validate_change(a,b)
    def test_exact_deterministic_positions(self):
        seq=['a','negative','b','a','a','negative','a'];p=slots(seq,'a',2,7)
        self.assertEqual(p,slots(seq,'a',2,7));self.assertEqual(len(set(p)),2)
        self.assertTrue(all(seq[i]=='a' for i in p))
    def test_insufficient_exposure_rejected(self):
        with self.assertRaises(ValueError):slots(['a'],'a',2,7)

if __name__=='__main__':unittest.main()
