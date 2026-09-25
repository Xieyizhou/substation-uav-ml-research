import unittest
import xml.etree.ElementTree as ET
from src.vision.canonical.hierarchy import flatten_equipment


class HierarchyTests(unittest.TestCase):
    def tree(self):
        return ET.fromstring('''<world><model name="station"><static>true</static><pose>-10 -10 0 0 0 0</pose>
          <model name="switch"><static>true</static><pose>10 2 0 0 0 0</pose><link name="link"><visual name="body"><geometry><box><size>1 2 3</size></box></geometry><material><diffuse>0.1 0.2 0.3 1</diffuse></material><plugin name="Label"><label>149</label></plugin></visual><collision name="body"><geometry><box><size>1 2 3</size></box></geometry></collision></link></model>
          <model name="floor"><static>true</static></model></model></world>''')
    def test_preserves_world_pose_and_all_link_content(self):
        world=self.tree();before=ET.tostring(world.find('./model/model/link'))
        changes=flatten_equipment(world,['switch'])
        moved=world.find('./model[@name="switch"]')
        self.assertEqual([float(v) for v in moved.findtext('pose').split()],[0,-8,0,0,0,0])
        self.assertEqual(before,ET.tostring(moved.find('link')))
        self.assertIsNotNone(world.find('./model[@name="station"]/model[@name="floor"]'))
        self.assertEqual(len(changes),1)
        self.assertEqual(flatten_equipment(world,['switch']),[])
    def test_rotated_parent_fails_before_mutation(self):
        world=self.tree();world.find('./model/pose').text='0 0 0 0 0 1'
        before=ET.tostring(world)
        with self.assertRaises(ValueError):flatten_equipment(world,['switch'])
        self.assertEqual(before,ET.tostring(world))
    def test_unknown_object_rejected(self):
        with self.assertRaises(ValueError):flatten_equipment(self.tree(),['missing'])
