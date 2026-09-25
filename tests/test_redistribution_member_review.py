import unittest
from unittest.mock import patch
from scripts.vision.review_redistribution_members import validate,object_sha256

class MemberReviewTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(event_id='P01',evidence_path='e',member={'image_path':'i'})
        self.e={'events':[self.row]}
        self.d=dict(event_id='P01',source_identity=object_sha256(self.row),reason='explicit',reviewed_at='2026-09-09',review_nature='AI辅助审核',training_approved=False,evidence_sha256='ok',image_sha256='ok')
    def test_missing_duplicate(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):validate(self.e,ds)
    def test_identity_stale(self):
        with self.assertRaises(ValueError):validate(self.e,[{**self.d,'source_identity':'old'}])
    def test_no_auto_admission(self):
        with self.assertRaises(ValueError):validate(self.e,[{**self.d,'training_approved':True}])
    def test_hash_checked(self):
        with patch('scripts.vision.review_redistribution_members.file_sha256',return_value='stale'):
            with self.assertRaises(ValueError):validate(self.e,[self.d])
    def test_screen_can_remain_nonadmitted(self):
        with patch('scripts.vision.review_redistribution_members.file_sha256',return_value='ok'):validate(self.e,[self.d])
