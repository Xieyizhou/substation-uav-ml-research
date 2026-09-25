from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from src.ml.artifacts import file_sha256
from scripts.vision.prepare_appearance_background_regression import copy_world


class AppearanceBackgroundPrepareTests(unittest.TestCase):
    def test_box_type_change_is_saved_on_the_mutated_tree(self):
        payload='''<sdf><world><scene><ambient>1</ambient><background>1</background></scene><light name="sun"><diffuse>1</diffuse></light><model name="substation_floor"><link><visual name="floor"><material><ambient>1</ambient><diffuse>1</diffuse></material></visual></link></model><model name="camera"><link><sensor type="boundingbox_camera"><camera><box_type>visible_2d</box_type></camera></sensor></link></model></world></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source"; target=root/"target"; source.mkdir()
            world=source/"world.sdf"; world.write_text(payload)
            plan={"files":{"world.sdf":file_sha256(world)},"objects":[]}
            with patch("scripts.vision.prepare_appearance_background_regression.read_record",return_value=plan):
                copy_world(source/"plan.json",target,"background")
            self.assertEqual(ET.parse(target/"world.sdf").findtext(".//camera/box_type"),"full_2d")


if __name__=="__main__": unittest.main()
