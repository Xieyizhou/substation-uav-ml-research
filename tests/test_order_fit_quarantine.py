import unittest
from scripts.vision.isolate_order_fit_quality_risks import classify
from scripts.vision.record_order_fit_risk_review import validate


class QuarantineTests(unittest.TestCase):
    def member(self,mid,line):
        return dict(member_id=mid,lineage_id=line,image_sha256='image',label_sha256='label')

    def test_same_lineage_and_history(self):
        ms=[self.member('a','shared'),self.member('b','shared'),self.member('c','other'),self.member('d','held')]
        r=classify(ms,dict(unresolved_content_members=['a'],limited_only_members=[]),{'d'})
        self.assertEqual([x['status'] for x in r],['quarantined','quarantined','requires_complete_quality_and_role_validation','quarantined'])
        self.assertTrue(all(not x['labels_removed'] and not x['training_eligible'] for x in r))

    def test_pending_never_approved(self):
        r=classify([self.member('a','x')],dict(unresolved_content_members=[],limited_only_members=['a']),set())
        self.assertEqual(r[0]['status'],'pending_limited_content_eligibility')

    def test_unknown_member(self):
        with self.assertRaises(ValueError):classify([],dict(unresolved_content_members=['x'],limited_only_members=[]),set())

    def test_missing_review(self):
        with self.assertRaises(ValueError):validate(dict(events=[dict(review_id='r')]),{})

    def test_duplicate_review(self):
        with self.assertRaises(ValueError):validate(dict(events=[dict(review_id='r'),dict(review_id='r')]),{'r':{}})
