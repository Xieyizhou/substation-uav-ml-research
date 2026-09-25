import unittest
import xml.etree.ElementTree as ET
from scripts.vision.closed_body_material_pilot import change_body,PALETTES
from scripts.vision.instance_visibility_diagnosis import add_sensor,validate_world
from tests.test_body_visibility_replay import fixture


def source():
    t=fixture();b=t.find(".//visual[@name='body']");m=ET.SubElement(b,'material')
    for k in ('ambient','diffuse','specular'):ET.SubElement(m,k).text='0.16 0.35 0.40 1'
    return t


class ClosedMaterial(unittest.TestCase):
    def test_closed_geometry_unchanged(self):
        for color in PALETTES.values():
            t=source();before=ET.tostring(t.getroot());d=change_body(t,'capacitor_east',color)
            self.assertEqual(before,ET.tostring(t.getroot()))
            for k in ('ambient','diffuse'):self.assertEqual(d.findtext(".//visual[@name='body']/material/"+k),color)
            self.assertEqual(d.findtext(".//visual[@name='body']/material/specular"),'0.16 0.35 0.40 1')
            self.assertEqual(len(d.findall('.//visual')),8);self.assertIsNotNone(d.find('.//collision'))
            validate_world(d,add_sensor(d))

    def test_extra_world_change_rejected(self):
        t=change_body(source(),'capacitor_east',PALETTES['warm']);d=add_sensor(t)
        d.find(".//visual[@name='base']").set('name','changed')
        with self.assertRaises(ValueError):validate_world(t,d)

    def test_missing_material_unsupported_pbr_and_alpha(self):
        for mode in ('missing','pbr','alpha'):
            t=source();m=t.find(".//visual[@name='body']/material")
            if mode=='missing':m.remove(m.find('diffuse'))
            elif mode=='pbr':ET.SubElement(m,'pbr')
            else:m.find('diffuse').text='1 1 1 .5'
            with self.assertRaises(ValueError):change_body(t,'capacitor_east',PALETTES['cool'])

    def test_same_palette_not_class_specific(self):
        t=source();t.find(".//model[@name='capacitor_east']").set('name','switchgear_west')
        for c in PALETTES.values():
            d=change_body(t,'switchgear_west',c)
            self.assertEqual(d.findtext(".//visual[@name='body']/material/diffuse"),c)


if __name__=='__main__':unittest.main()
