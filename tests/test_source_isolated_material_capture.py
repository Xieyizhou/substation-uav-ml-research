import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.vision import source_isolated_material_capture as m
from src.vision.canonical.plan import bounds


class SourceIsolatedCaptureTests(unittest.TestCase):
    def test_frozen_matrix_and_roles(self):
        p=m.freeze()
        self.assertEqual(len(p['units']),64)
        self.assertEqual(sum(u['pilot'] for u in p['units']),16)
        self.assertEqual(len({u['pair_id'] for u in p['units']}),16)
        self.assertFalse(p['training_admitted'])
        self.assertFalse(p['promotable'])
        self.assertEqual(p['independent_asset_families_added'],0)

    def test_complete_mapping_config_and_material_only(self):
        p=m.freeze()
        for layout in p['layouts']:
            lid=layout['layout_id']; base=m.OUT/'plans'/lid/'original'
            plan=m.read_record(base/'plan.json');cfg=m.json.loads((base/'obstacles.json').read_text())
            objects={o['name']:o for o in plan['objects']}
            for obj in cfg['obstacles']:
                self.assertEqual(bounds(obj,cfg['gazebo_world_origin_m']),objects[obj['name']]['bounds'])
            m.validate_preflight(plan,base,plan['calibration_views'])
            original=ET.parse(base/'world.sdf').getroot()
            names={o['name'] for o in plan['objects'] if o['category'] in m.COUNTS}
            for variant in ('warm','cool','neutral'):
                tree=ET.parse(m.OUT/'plans'/lid/variant/'world.sdf').getroot()
                self.assertTrue(m.check_only_materials(original,tree,names))
                broken=copy.deepcopy(tree)
                broken.find('./world/model/pose').text='100 100 100 0 0 0'
                with self.assertRaises(ValueError):m.check_only_materials(original,broken,names)

    def test_packed_targets_do_not_overlap(self):
        for layout in m.freeze()['layouts']:
            plan=m.read_record(m.OUT/'plans'/layout['layout_id']/'original/plan.json')
            for obj in plan['objects']:
                if obj['category'] not in m.COUNTS:continue
                for other in plan['objects']:
                    if obj['name']==other['name']:continue
                    self.assertFalse(m.overlap(obj['bounds'],other['bounds']))


if __name__=='__main__':unittest.main()
