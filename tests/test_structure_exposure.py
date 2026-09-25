import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.vision.audit_structure_exposure import label_counts,pixels

class CensusTest(unittest.TestCase):
    def test_full_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.txt';p.write_text('0 .5 .5 .2 .2\n0 .4 .4 .1 .1\n1 .3 .3 .2 .2\n')
            self.assertEqual(label_counts(p,['reactor','other']),{'reactor':2,'other':1})
            p.write_text('');self.assertEqual(label_counts(p,['reactor']),{})
    def test_bad_label(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.txt'
            for text in ('0.5 .5 .5 .1 .1','4 .5 .5 .1 .1','0 .5 .5 -1 .1','0 nan .5 .1 .1'):
                p.write_text(text)
                with self.assertRaises(ValueError):label_counts(p,['reactor'])
    def test_lossless_container_change(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.png';b=Path(d)/'b.ppm'
            im=Image.new('RGB',(12,9),'gray');im.save(a);im.save(b)
            self.assertEqual(pixels(a),pixels(b))
            im.putpixel((0,0),(0,0,0));im.save(b)
            self.assertNotEqual(pixels(a),pixels(b))
