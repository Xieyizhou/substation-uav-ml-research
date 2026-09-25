import unittest
from unittest.mock import patch
from pathlib import Path
from scripts.vision.expand_material_view_n05 import mask_gate

class ExpansionTests(unittest.TestCase):
    def good(self):return {'stable_frames':3,'records':[{'maximum_box_delta_px':0}]*3}
    def test_exact_mask(self):
        with patch('scripts.vision.expand_material_view_n05.prior.file_sha256',return_value='same'):
            mask_gate(self.good(),Path('/evidence'),Path('/reference'))
    def test_changed_mask(self):
        with patch('scripts.vision.expand_material_view_n05.prior.file_sha256',side_effect=['different','reference']):
            with self.assertRaises(ValueError):mask_gate(self.good(),Path('/evidence'),Path('/reference'))
    def test_box_drift(self):
        r=self.good();r['records']=[{'maximum_box_delta_px':1.01}]
        with self.assertRaises(ValueError):mask_gate(r,Path('/evidence'),Path('/reference'))
    def test_unstable(self):
        r=self.good();r['stable_frames']=2
        with self.assertRaises(ValueError):mask_gate(r,Path('/evidence'),Path('/reference'))
