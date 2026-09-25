import copy,unittest
import xml.etree.ElementTree as ET
from scripts.vision.cool_light_capture import LIGHT,lighting_only
class LightingGate(unittest.TestCase):
    def setUp(self):
        self.root=ET.fromstring('<sdf><world><scene><ambient>1 1 1 1</ambient></scene><light name="sun"><diffuse>1 1 1 1</diffuse></light><model name="target"><pose>0 0 0 0 0 0</pose></model></world></sdf>')
        self.changed=copy.deepcopy(self.root)
        for path,value in LIGHT.items():self.changed.find(path).text=value
    def test_only_lighting(self):lighting_only(self.root,self.changed)
    def test_geometry(self):
        self.changed.find('.//pose').text='1 0 0 0 0 0'
        with self.assertRaises(ValueError):lighting_only(self.root,self.changed)
    def test_wrong_color(self):
        self.changed.find('.//ambient').text='0.42 0.42 0.42 1'
        with self.assertRaises(ValueError):lighting_only(self.root,self.changed)
    def test_missing(self):
        self.changed.find('.//scene').remove(self.changed.find('.//ambient'))
        with self.assertRaises(ValueError):lighting_only(self.root,self.changed)
    def test_duplicate(self):
        self.changed.find('.//scene').append(copy.deepcopy(self.changed.find('.//ambient')))
        with self.assertRaises(ValueError):lighting_only(self.root,self.changed)
if __name__=='__main__':unittest.main()
