import copy,unittest
import xml.etree.ElementTree as ET
from scripts.vision.physical_low_light_capture import LIGHT,freeze,lighting_only
from scripts.vision.verify_physical_low_light import verify_capture

class LowLightTests(unittest.TestCase):
    def test_only_lighting(self):
        a=ET.fromstring('<sdf><world><scene><ambient>old</ambient><background>bg</background></scene><light name="sun"><diffuse>old</diffuse></light><model name="body"/></world></sdf>')
        b=copy.deepcopy(a)
        for p,v in LIGHT.items():b.find(p).text=v
        lighting_only(a,b)
        b.find('.//background').text='changed'
        with self.assertRaises(ValueError):lighting_only(a,b)

    def test_missing_light(self):
        a=ET.fromstring('<sdf><world/></sdf>')
        with self.assertRaises(ValueError):lighting_only(a,a)

    def test_real_pilot_mapping_and_frame_gates(self):
        p=freeze()
        for u in p['units']:
            if u['unit_id'] not in p['pilot_units']:continue
            r,m,_=verify_capture(u)
            self.assertTrue(all(isinstance(k,str) for k in m))
            self.assertEqual(len(m),len({v['object_id'] for v in m.values()}))
            self.assertEqual(r['status'],'captured')

if __name__=='__main__':unittest.main()
