import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.vision.neutral_gray_capture import material_nodes,material_only

class GrayWorld(unittest.TestCase):
    def setUp(self):
        self.a=ET.fromstring('<sdf><world><scene><ambient>.38 .42 .48 1</ambient></scene><model name="target"><link><visual name="body"><geometry><box><size>1 1 1</size></box></geometry><material><ambient>.42 .42 .42 1</ambient><diffuse>.42 .42 .42 1</diffuse></material><plugin><label>201</label></plugin></visual><visual name="base"><material><diffuse>.1 .1 .1 1</diffuse></material></visual></link></model></world></sdf>')
        self.b=copy.deepcopy(self.a)
        for n in material_nodes(self.b):n.text='.35 .35 .35 1'
    def test_allowed(self):material_only(self.a,self.b)
    def test_geometry(self):
        self.b.find('.//size').text='2 1 1'
        with self.assertRaises(ValueError):material_only(self.a,self.b)
    def test_lighting(self):
        self.b.find('.//scene/ambient').text='1 1 1 1'
        with self.assertRaises(ValueError):material_only(self.a,self.b)
    def test_base(self):
        self.b.find('.//visual[@name="base"]/material/diffuse').text='.35 .35 .35 1'
        with self.assertRaises(ValueError):material_only(self.a,self.b)
    def test_mapping(self):
        self.b.find('.//label').text='202'
        with self.assertRaises(ValueError):material_only(self.a,self.b)
    def test_wrong_value(self):
        material_nodes(self.b)[0].text='.36 .36 .36 1'
        with self.assertRaises(ValueError):material_only(self.a,self.b)
    def test_duplicate_field(self):
        self.b.find('.//visual/material').append(copy.deepcopy(self.b.find('.//visual/material/ambient')))
        with self.assertRaises(ValueError):material_only(self.a,self.b)

if __name__=='__main__':unittest.main()
