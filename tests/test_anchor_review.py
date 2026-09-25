import unittest,copy
from unittest.mock import patch
from scripts.vision.record_anchor_review import OUT,read,validate

class AnchorReviewTests(unittest.TestCase):
    def setUp(self):
        self.m=read(OUT/'manifest.json');self.d=read(OUT/'visual-review.json')['decisions']
    def test_complete(self):validate(self.m,self.d)
    def test_missing_duplicate_unknown(self):
        for mode in ('missing','duplicate','unknown'):
            d=copy.deepcopy(self.d)
            if mode=='missing':d.pop()
            if mode=='duplicate':d[-1]=d[0]
            if mode=='unknown':d[0]['content_category']='unknown'
            with self.assertRaises(ValueError):validate(self.m,d)
    def test_stale_and_box_change(self):
        d=copy.deepcopy(self.d);d[0]['confidence']+=.01
        with self.assertRaises(ValueError):validate(self.m,d)
        with patch('scripts.vision.record_anchor_review.file_sha256',return_value='stale'):
            with self.assertRaises(ValueError):validate(self.m,self.d)
