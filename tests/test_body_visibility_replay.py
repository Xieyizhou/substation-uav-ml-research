import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.vision.run_body_visibility_control import select
from scripts.vision.run_body_visibility_intervention import without_body,modified_sensor,validate_modified
from tests.test_instance_visibility_diagnosis import WORLD


def fixture():
    tree=ET.ElementTree(ET.fromstring(WORLD));m=ET.SubElement(tree.find('.//world'),'model',name='capacitor_east');link=ET.SubElement(m,'link',name='link')
    ET.SubElement(link,'collision',name='collision')
    for name in ['body','base',*[f'capacitor_{i}' for i in range(6)]]:ET.SubElement(link,'visual',name=name)
    return tree


class BodyReplay(unittest.TestCase):
    def test_removes_visual_only_preserves_source(self):
        tree=fixture();before=ET.tostring(tree.getroot());derived=modified_sensor(tree)
        validate_modified(tree,derived)
        self.assertEqual(ET.tostring(tree.getroot()),before)
        self.assertIsNotNone(derived.find('.//collision'))
        self.assertIsNone(derived.find(".//visual[@name='body']"))
        self.assertEqual(len(derived.findall('.//visual')),7)

    def test_other_changes_rejected(self):
        tree=fixture()
        for xpath in ('.//collision',".//visual[@name='base']",".//visual[@name='capacitor_0']",'.//world'):
            derived=modified_sensor(tree);derived.find(xpath).set('name','changed')
            with self.assertRaises(ValueError):validate_modified(tree,derived)

    def test_sensor_misalignment_rejected(self):
        tree=fixture();derived=modified_sensor(tree)
        derived.find(".//sensor[@name='diagnostic_instances']/camera/horizontal_fov").text='1'
        with self.assertRaises(ValueError):validate_modified(tree,derived)

    def test_missing_duplicate_body_or_cylinder_rejected(self):
        tree=fixture();link=tree.find(".//model[@name='capacitor_east']/link")
        link.remove(link.find("visual[@name='body']"))
        with self.assertRaises(ValueError):without_body(tree)
        tree=fixture();link=tree.find(".//model[@name='capacitor_east']/link")
        ET.SubElement(link,'visual',name='body')
        with self.assertRaises(ValueError):without_body(tree)
        tree=fixture();link=tree.find(".//model[@name='capacitor_east']/link");link.remove(link.find("visual[@name='capacitor_0']"))
        with self.assertRaises(ValueError):without_body(tree)

    def test_selection_ignores_model_score_and_excludes_gaps(self):
        base=dict(class_name='capacitor_bank',gaps=[],actual_exposures={'7':1,'17':1,'27':1})
        rows=[dict(base,member_id='b',review_id='b'),dict(base,member_id='a',review_id='a')]
        rv={'decisions':[dict(review_id=i,decision='condition_recorded',occlusion_condition='no_obvious_foreground_occlusion') for i in ('a','b')]}
        self.assertEqual(select({'training':rows},rv)['member_id'],'a')
        rows[1]['gaps']=['unknown'];self.assertEqual(select({'training':rows},rv)['member_id'],'b')
        rows[0]['actual_exposures']={'7':0,'17':0,'27':0}
        with self.assertRaises(ValueError):select({'training':rows},rv)


if __name__=='__main__':unittest.main()
