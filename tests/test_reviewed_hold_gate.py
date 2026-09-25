import unittest
from scripts.vision.reviewed_hold_gate import validate_review


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        label={'runtime_label':'1'}
        self.e={'identity':'e','events':[{'member_id':'m','labels':[label],'page_sha256':'p'}]}
        self.d=dict(member_id='m',runtime_label='1',evidence_identity='e',page_sha256='p',label=label,status='identifiable_content_with_recorded_limits',reason='observed',review_nature='AI-assisted',reviewed_at='date')

    def test_valid(self):validate_review(self.e,{'decisions':[self.d],'new_risk_members':[]})

    def test_missing_duplicate_stale(self):
        for ds in ([],[self.d,self.d],[dict(self.d,page_sha256='stale')],[dict(self.d,status='unknown')]):
            with self.assertRaises(ValueError):validate_review(self.e,{'decisions':ds,'new_risk_members':[]})

    def test_new_risk_stops(self):
        with self.assertRaises(ValueError):validate_review(self.e,{'decisions':[self.d],'new_risk_members':['m']})
