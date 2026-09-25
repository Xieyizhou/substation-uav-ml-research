import unittest
from scripts.vision.review_redistribution_occlusion import validate_decisions, OCCLUDED

class ReviewTests(unittest.TestCase):
    def rows(self):
        return [dict(event_id=e, reason='Explicit observation',review_nature='AI辅助审核',training_approved=False,
            replay_status='original_pixel_evidence_certified',visible_pixel_count=1) for e in sorted(OCCLUDED)]
    def test_complete(self): validate_decisions(self.rows())
    def test_missing_and_duplicate(self):
        ds=self.rows()
        for rows in (ds[:-1],ds[:-1]+[ds[0]]):
            with self.assertRaises(ValueError):validate_decisions(rows)
    def test_no_certification_from_empty_unaligned_or_admission(self):
        for k,v in [('visible_pixel_count',0),('replay_status','alignment_held'),('training_approved',True),('reason','')]:
            ds=self.rows();ds[0][k]=v
            with self.assertRaises(ValueError):validate_decisions(ds)
