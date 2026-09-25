import unittest
from scripts.vision.audit_lineage_capped_results import verify_decisions

class ResultReviewTests(unittest.TestCase):
    def setUp(self):
        self.e={'events':[{'event_id':'FP01','kind':'FP','predictions':[{'confidence':.4}], 'source':{'image_sha256':'im'},'evidence_sha256':'ev','crops':[{'sha256':'crop'}]}]}
        self.d=dict(review_id='FP01:0',image_sha256='im',evidence_sha256='ev',crop_sha256='crop',reason='reviewed structure',review_nature='AI辅助审核',reviewed_at='now',pixel_visibility_certified=False,prediction={'confidence':.4})
    def test_complete(self):verify_decisions(self.e,[self.d])
    def test_missing_duplicate(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):verify_decisions(self.e,ds)
    def test_stale_or_changed_prediction(self):
        for changes in [dict(image_sha256='bad'),dict(evidence_sha256='bad'),dict(crop_sha256='bad'),dict(prediction={'confidence':.5}),dict(reason=''),dict(pixel_visibility_certified=True)]:
            with self.assertRaises(ValueError):verify_decisions(self.e,[{**self.d,**changes}])
