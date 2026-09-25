import unittest
from unittest.mock import patch
from scripts.vision.finalize_reviewed_hold import validate_errors


class ErrorReviewTests(unittest.TestCase):
    def setUp(self):
        self.e={'identity':'e','events':[{'event_id':'FP01','kind':'FP','predictions':[{}],'page_sha256':'p','page':'/p.png'}]}
        self.d=dict(decision_id='FP01-0',page_sha256='p',review_nature='AI-assisted',reason='structure',reviewed_at='date',status='reviewed')

    def test_no_automatic_pass(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):validate_errors(self.e,dict(evidence_identity='e',decisions=ds))

    def test_unknown_retained(self):
        with patch('scripts.vision.finalize_reviewed_hold.prior.file_sha256',return_value='p'):
            self.assertFalse(validate_errors(self.e,dict(evidence_identity='e',decisions=[dict(self.d,status='unknown')])))

    def test_stale_rejected(self):
        with self.assertRaises(ValueError):validate_errors(self.e,dict(evidence_identity='old',decisions=[self.d]))
