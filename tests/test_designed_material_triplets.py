import unittest
import xml.etree.ElementTree as ET
from scripts.vision.capture_designed_material_triplets import OUT,prior
from scripts.vision.build_material_view_world_drafts import check_only_materials

class FrozenTripletTests(unittest.TestCase):
    def test_frozen_eight_units_and_unique_targets(self):
        p=prior.read(OUT/'protocol.json');prior.verify(p)
        self.assertEqual(len(p['units']),8)
        self.assertEqual({r['key'] for r in p['units']},{f'G{i:02}/{v}' for i in range(1,5) for v in ('warm','cool')})
        self.assertEqual(len({r['target_object_id'] for r in p['units']}),4)
    def test_actual_xml_only_target_body_changed(self):
        p=prior.read(OUT/'protocol.json')
        for r in p['units']:
            a=ET.parse(r['frame']['source_world']).getroot();b=ET.parse(r['world']).getroot()
            changed=check_only_materials(a,b,{r['target_object_id']},('body','reactor'))
            self.assertEqual(len(changed),2)
    def test_sources_are_four_categories(self):
        p=prior.read(OUT/'protocol.json')
        self.assertEqual({r['frame']['class_name'] for r in p['units']},{'transformer','switchgear','capacitor_bank','reactor'})
