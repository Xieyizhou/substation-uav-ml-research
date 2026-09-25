import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.vision.trace_l05_risk_scope import pixels

class PixelIdentityTests(unittest.TestCase):
    def test_lossless_format_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.png',Path(d)/'b.ppm'
            im=Image.new('RGB',(7,5),(10,20,30));im.save(a);im.save(b)
            self.assertEqual(pixels(a),pixels(b))

    def test_one_pixel_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.png',Path(d)/'b.png'
            im=Image.new('RGB',(7,5));im.save(a);im.putpixel((0,0),(1,0,0));im.save(b)
            self.assertNotEqual(pixels(a),pixels(b))

    def test_dimensions_bound_even_equal_rgb_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.png',Path(d)/'b.png'
            Image.new('RGB',(7,5)).save(a);Image.new('RGB',(5,7)).save(b)
            self.assertNotEqual(pixels(a),pixels(b))
