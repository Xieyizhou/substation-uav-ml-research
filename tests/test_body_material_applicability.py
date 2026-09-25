import unittest
import xml.etree.ElementTree as ET
from scripts.vision.test_body_material_applicability import bounds,strictly_inside,signature,model_mapping


class BodyGeometry(unittest.TestCase):
    def visual(self,geometry,pose='0 0 0 0 0 0'):
        return ET.fromstring(f'<visual><pose>{pose}</pose><geometry>{geometry}</geometry></visual>')

    def test_exact_containment_clearance(self):
        b=bounds(self.visual('<box><size>2.58 2.58 1.62</size></box>','0 0 .9 0 0 0'))
        c=bounds(self.visual('<cylinder><radius>.1</radius><length>1.17</length></cylinder>','-.75 -.66 .98 0 0 0'))
        r=strictly_inside(b,c)
        self.assertTrue(r['strictly_inside']);self.assertAlmostEqual(r['min_clearance_m'],.145)

    def test_touching_not_strictly_contained(self):
        self.assertFalse(strictly_inside([[0,0,0],[1,1,1]],[[0,.2,.2],[.8,.8,.8]])['strictly_inside'])

    def test_protruding_control(self):
        self.assertFalse(strictly_inside([[0,0,0],[1,1,1]],[[.2,.2,.2],[.8,.8,1.1]])['strictly_inside'])

    def test_rotation_relative_frame_and_invalid_dimensions_rejected(self):
        for v in [self.visual('<box><size>2 2 2</size></box>','0 0 0 0 0 .1'),
                  self.visual('<box><size>-2 2 2</size></box>'),
                  self.visual('<box><size>nan 2 2</size></box>')]:
            with self.assertRaises(ValueError):bounds(v)
        v=self.visual('<box><size>2 2 2</size></box>');v.find('pose').set('relative_to','other')
        with self.assertRaises(ValueError):bounds(v)

    def test_missing_body_not_silently_accepted(self):
        with self.assertRaises(ValueError):signature(ET.fromstring('<model><link/></model>'))

    def test_opaque_required_and_six_cylinders(self):
        m=ET.fromstring('<model><link><visual name="body"><geometry><box><size>4 4 4</size></box></geometry><material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material></visual></link></model>')
        for i in range(6):
            v=self.visual('<cylinder><radius>.1</radius><length>1</length></cylinder>');v.set('name',f'capacitor_{i}');m.find('link').append(v)
        self.assertTrue(signature(m)['six_cylinders_strictly_enclosed'])
        ET.SubElement(m.find('link/visual'),'transparency').text='.5'
        self.assertFalse(signature(m)['six_cylinders_strictly_enclosed'])

    def test_collision_in_saved_visual_labels(self):
        root=ET.fromstring('<sdf><world><model name="a"><link><visual><plugin name="gz::sim::systems::Label"><label>127</label></plugin></visual></link></model><model name="b"><link><visual><plugin name="gz::sim::systems::Label"><label>0127</label></plugin></visual></link></model></world></sdf>')
        with self.assertRaises(ValueError):model_mapping(root)


if __name__=='__main__':unittest.main()
