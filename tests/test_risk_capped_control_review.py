import unittest
from scripts.vision.decide_risk_capped_control import validate
from scripts.vision.finish_risk_capped_control import require_quality_ready

class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.e={'events':[{'labels':[{'event_id':'box','crop_sha256':'hash'}]}]}
        self.d={'event_id':'box','crop_sha256':'hash','reason':'observed','pixel_visibility_certified':False,'training_approved':False}
    def test_explicit_review(self):validate(self.e,[self.d])
    def test_missing_duplicate(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):validate(self.e,ds)
    def test_stale_or_unjustified(self):
        for k,v in [('crop_sha256','stale'),('reason',''),('training_approved',True),('pixel_visibility_certified',True)]:
            with self.assertRaises(ValueError):validate(self.e,[{**self.d,k:v}])
    def test_second_round_risk_stops(self):
        with self.assertRaises(ValueError):require_quality_ready({'decisions':[self.d]},{'decisions':[{'status':'insufficient_or_uncertain'}]},dict(additional_resolves_used=1,additional_resolves_remaining=0))
    def test_budget_cannot_expand(self):
        with self.assertRaises(ValueError):require_quality_ready({'decisions':[self.d]},{'decisions':[{'status':'identifiable_geometry_with_recorded_limits'}]},dict(additional_resolves_used=2,additional_resolves_remaining=0))
