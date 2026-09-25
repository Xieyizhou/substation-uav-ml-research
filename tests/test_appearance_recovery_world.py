import unittest
import xml.etree.ElementTree as ET
from scripts.vision.capture_appearance_recovery_pilot import transform,payload

class WorldTests(unittest.TestCase):
    def test_exact_changes(self):
        root=ET.fromstring('<sdf><world><scene><ambient>1</ambient></scene><light name="sun"><diffuse>1</diffuse></light><model name="t"><link><visual name="body"><material><ambient>old</ambient><diffuse>old</diffuse></material><geometry>unchanged</geometry></visual></link></model><sensor type="boundingbox_camera"><camera><box_type>visible_2d</box_type></camera></sensor></world></sdf>')
        m={'materials':{'steel':'new'},'lights':{'warm_dim':{'ambient':'dim','sun_diffuse':'warm'}}}
        transform(root,'steel_warm_dim',m,[dict(name='t',category='transformer')])
        self.assertEqual(root.findtext('.//box_type'),'full_2d')
        self.assertEqual(root.findtext('.//geometry'),'unchanged')
        self.assertEqual(root.findtext('.//material/ambient'),'new')
        old=payload(root);root.find('.//geometry').text='changed';self.assertNotEqual(old,payload(root))
    def test_missing_sensor(self):
        with self.assertRaises(ValueError):transform(ET.fromstring('<sdf><world/></sdf>'),'original',{},[])

if __name__=='__main__':unittest.main()
