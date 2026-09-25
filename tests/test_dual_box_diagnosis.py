import copy
import unittest
import xml.etree.ElementTree as ET
from tests.test_instance_visibility_diagnosis import WORLD
from scripts.vision.run_dual_box_diagnosis import dual_sensor,validate_dual,parse_visible


def source():
    tree=ET.ElementTree(ET.fromstring(WORLD));link=tree.find('.//link')
    box=copy.deepcopy(link.find('sensor'));box.set('name','research_boxes');box.set('type','boundingbox_camera')
    box.find('topic').text='/research_camera/boxes';ET.SubElement(box.find('camera'),'box_type').text='full_2d';link.append(box)
    return tree


class DualMode(unittest.TestCase):
    def test_existing_sensor_unchanged(self):
        original=source();raw=ET.tostring(original.getroot());derived=dual_sensor(original);validate_dual(original,derived)
        self.assertEqual(ET.tostring(original.getroot()),raw)
        self.assertEqual(derived.findtext(".//sensor[@name='research_boxes']/camera/box_type"),'full_2d')
        self.assertEqual(derived.findtext(".//sensor[@name='diagnostic_visible_boxes']/camera/box_type"),'visible_2d')

    def test_mismatched_geometry_resolution_or_type_rejected(self):
        original=source()
        for path in ('camera/horizontal_fov','camera/image/width','camera/box_type','topic'):
            derived=dual_sensor(original);derived.find(".//sensor[@name='diagnostic_visible_boxes']/"+path).text='changed'
            with self.assertRaises(ValueError):validate_dual(original,derived)
        derived=dual_sensor(original);derived.find('.//world').set('name','changed')
        with self.assertRaises(ValueError):validate_dual(original,derived)

    def test_missing_or_duplicate_sensor_rejected(self):
        original=source();derived=dual_sensor(original);link=derived.find('.//link');node=link.find("sensor[@name='diagnostic_visible_boxes']")
        link.append(copy.deepcopy(node))
        with self.assertRaises(ValueError):validate_dual(original,derived)
        link.remove(node);link.remove(link.find("sensor[@name='diagnostic_visible_boxes']"))
        with self.assertRaises(ValueError):validate_dual(original,derived)

    def test_visible_pixel_extrema_not_full_box_semantics(self):
        item=dict(label=128,box=dict(minCorner=dict(x=59,y=1052),maxCorner=dict(x=305,y=1079)))
        self.assertEqual(parse_visible({'annotatedBox':[item]},{'128':{}})['128']['half_open'],[59,1052,306,1080])
        with self.assertRaises(ValueError):parse_visible({'annotatedBox':[item,item]},{'128':{}})
        with self.assertRaises(ValueError):parse_visible({'annotatedBox':[item]},{'113':{}})


if __name__=='__main__':unittest.main()
