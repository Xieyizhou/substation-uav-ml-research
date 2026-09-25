import unittest
import json
import xml.etree.ElementTree as ET
from scripts.vision import closed_exterior_material_capture as m


class ClosedExteriorCaptureTests(unittest.TestCase):
    def test_counts_and_fronts(self):
        p=m.freeze();self.assertEqual(len(p['units']),64);self.assertEqual(len(p['runs']),8)
        for layout in p['layouts']:
            views=layout['selected'];self.assertEqual(len(views),8)
            sw=[v for v in views if v['category']=='switchgear']
            self.assertIn(sw[0]['bearing'],(255,270,285))
            self.assertIn(sw[1]['bearing'],(210,225,240,300,315,330))
            self.assertEqual(len({v['distance'] for v in views if v['category']=='reactor'}),3)
            self.assertEqual(len({v['distance'] for v in views if v['category']=='capacitor_bank'}),2)
        self.assertFalse(p['training_ready']);self.assertFalse(p['training_started'])

    def test_worlds_and_gates(self):
        p=m.freeze()
        for run in p['runs']:
            plan=m.read_record(run['plan_path']);folder=m.Path(run['plan_path']).parent
            for name,h in plan['files'].items():self.assertEqual(m.prior.file_sha256(folder/name),h)
            m.validate_preflight(plan,folder,plan['calibration_views'])
            original=ET.parse(folder.parent/'original/world.sdf').getroot()
            world=ET.parse(folder/'world.sdf').getroot()
            names={o['name'] for o in plan['objects'] if o['category'] in m.old.COUNTS}
            changes=m.old.check_only_materials(original,world,names)
            self.assertEqual(bool(changes),run['variant']!='original')


if __name__=='__main__':unittest.main()
