import unittest
from unittest.mock import patch
from scripts.vision.review_whole_image_hold_errors import validate

class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.e={'focus_ids':['L001'],'negative':[]}
        self.d=dict(review_id='L001',reason='explicit observation',review_nature='AI辅助审核',reviewed_at='2026-09-09',
                    image_path='a',image_sha256='ok',evidence_path='b',evidence_sha256='ok',pixel_visibility_certified=False)
    def test_missing_duplicate_reject(self):
        for ds in [[],[self.d,self.d]]:
            with self.assertRaises(ValueError):validate(self.e,ds)
    def test_stale_reject(self):
        with patch('scripts.vision.review_whole_image_hold_errors.file_sha256',return_value='stale'):
            with self.assertRaises(ValueError):validate(self.e,[self.d])
    def test_no_mask_certification(self):
        with patch('scripts.vision.review_whole_image_hold_errors.file_sha256',return_value='ok'):
            validate(self.e,[self.d])
            with self.assertRaises(ValueError):validate(self.e,[{**self.d,'pixel_visibility_certified':True}])
