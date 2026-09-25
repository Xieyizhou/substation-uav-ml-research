import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.vision.build_material_view_world_drafts import valid_clock,check_only_materials


class DraftTests(unittest.TestCase):
    def test_protobuf_seconds(self):
        for sec in (1,'1',20,'20'):
            self.assertTrue(valid_clock({'header':{'stamp':{'sec':sec,'nsec':'0'}}}))

    def test_bad_clocks(self):
        for stamp in ({},{'sec':'nan'},{'sec':'1.2'},{'sec':'0'},{'sec':1,'nsec':1000000000}):
            self.assertFalse(valid_clock({'header':{'stamp':stamp}}))

    def tree(self):
        return ET.fromstring('<sdf><world><model name="x"><pose>0 0 0</pose><link name="l"><visual name="body"><geometry><box/></geometry><material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse></material></visual></link></model></world></sdf>')

    def test_material_allowed(self):
        a=self.tree();b=copy.deepcopy(a);b.find('.//ambient').text='1 1 1 1'
        self.assertEqual(len(check_only_materials(a,b,{'x'})),1)

    def test_geometry_pose_or_other_model_rejected(self):
        for path,text in (('.//pose','1 0 0'),('.//geometry/box','changed')):
            a=self.tree();b=copy.deepcopy(a);b.find(path).text=text
            with self.assertRaises(ValueError):check_only_materials(a,b,{'x'})

    def test_removed_field_rejected(self):
        a=self.tree();b=copy.deepcopy(a);b.find('.//material').remove(b.find('.//ambient'))
        with self.assertRaises(ValueError):check_only_materials(a,b,{'x'})
