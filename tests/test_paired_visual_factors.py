import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from scripts.vision.prepare_paired_visual_factors import assert_allowed_world_diff


WORLD='''<sdf><world><scene><ambient>1</ambient><background>2</background></scene><light name="sun"><diffuse>3</diffuse></light><model name="thing"><link><visual name="body"><material><ambient>4</ambient><diffuse>5</diffuse></material></visual></link></model></world></sdf>'''


class WorldDiffTests(unittest.TestCase):
    def test_non_allowed_change_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"a.sdf"; b=Path(directory)/"b.sdf"
            tree=ET.ElementTree(ET.fromstring(WORLD)); ET.indent(tree,space="  "); tree.write(a)
            tree=ET.ElementTree(ET.fromstring(WORLD)); tree.find(".//model").set("name","changed"); ET.indent(tree,space="  "); tree.write(b)
            with self.assertRaisesRegex(ValueError,"Non-allowed"):
                assert_allowed_world_diff(a,b,"lighting")

    def test_allowed_lighting_change_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/"a.sdf"; b=Path(directory)/"b.sdf"
            one=ET.ElementTree(ET.fromstring(WORLD)); ET.indent(one,space="  "); one.write(a)
            two=ET.ElementTree(ET.fromstring(WORLD)); two.find(".//scene/ambient").text="9"; two.find(".//light/diffuse").text="8"; ET.indent(two,space="  "); two.write(b)
            assert_allowed_world_diff(a,b,"lighting")


if __name__=="__main__": unittest.main()
