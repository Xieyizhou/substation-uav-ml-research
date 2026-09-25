import unittest
from unittest.mock import patch
from scripts.vision.record_multiscale_error_review import validate
from scripts.vision.diagnose_multiscale_errors import truth_key

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.e={'events':[{'event_id':'E01','kind':'LOSS','source':{'image_sha256':'im'},'page_sha256':'page','page_path':'x'}]}
        self.d={'decision_id':'E01:0','image_sha256':'im','evidence_sha256':'page','reason':'explicit observation','review_nature':'AI-assisted','reviewed_at':'2026-09-11'}
    def test_missing(self):
        with self.assertRaises(ValueError):validate(self.e,[])
    def test_duplicate(self):
        with self.assertRaises(ValueError):validate(self.e,[self.d,self.d])
    def test_stale(self):
        with patch('scripts.vision.record_multiscale_error_review.prior.file_sha256',return_value='changed'):
            with self.assertRaises(ValueError):validate(self.e,[self.d])
    def test_valid(self):
        with patch('scripts.vision.record_multiscale_error_review.prior.file_sha256',return_value='page'):
            validate(self.e,[self.d])
    def test_identity_uses_coordinates_not_index(self):
        a={'class_name':'switchgear','bbox_xyxy':[1,2,3,4]}
        b={'class_name':'switchgear','bbox_xyxy':[2,2,3,4]}
        self.assertNotEqual(truth_key(a),truth_key(b))

if __name__=='__main__':unittest.main()
