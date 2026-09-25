import unittest
from scripts.vision.review_full_image_appearance import box_delta
from scripts.vision.audit_full_image_pilot_v2 import pixels
from scripts.vision.verify_pixel_duplicates import rgb_digest
from PIL import Image
from pathlib import Path
import tempfile

class AppearanceTests(unittest.TestCase):
    def test_pair_delta(self):
        self.assertEqual(box_delta({'a':[0,0,4,4]},{'a':[0,0,5,4]}),1)
        self.assertGreater(box_delta({'a':[0,0,4,4]},{'a':[0,0,6,4]}),1)
    def test_wrong_instance(self):
        with self.assertRaises(ValueError):box_delta({'a':[0,0,4,4]},{'b':[0,0,4,4]})
    def test_pixel_cache_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.png';q=Path(d)/'a.ppm'
            Image.new('RGB',(3,2),(10,20,30)).save(p);Image.open(p).save(q)
            self.assertEqual(pixels(p),pixels(q))
            self.assertEqual(pixels(p),rgb_digest(p.read_bytes())[0])
