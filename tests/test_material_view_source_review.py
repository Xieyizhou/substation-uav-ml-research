import copy
import unittest
from scripts.vision.finalize_material_view_source_review import validate_observations, summarize_frames


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.r = {'events':[{'event_id':'N01-00','source_review_id':'N01'}],
                  'sources':[{'source_review_id':'N01','source_member_id':'a','lineage_id':'p'}]}
        self.n = {'source_review_sha256':'hash','review_type':'AI辅助审核','reviewed_at':'2026-09-11',
                  'decisions':[['N01-00','pending','仅有片段']], 'frame_checks':{'N01':'checked'}}

    def test_pending_holds_whole_frame(self):
        d=validate_observations(self.r,self.n,'hash')
        self.assertEqual(summarize_frames(self.r,self.n,d)[0]['status'],'held_for_evidence')

    def test_missing_and_duplicate_decisions(self):
        for ds in ([],self.n['decisions']*2):
            n=copy.deepcopy(self.n);n['decisions']=ds
            with self.assertRaises(ValueError):validate_observations(self.r,n,'hash')

    def test_stale_review(self):
        with self.assertRaises(ValueError):validate_observations(self.r,self.n,'changed')

    def test_missing_frame_review(self):
        n=copy.deepcopy(self.n);n['frame_checks']={}
        with self.assertRaises(ValueError):validate_observations(self.r,n,'hash')

    def test_observed_content_is_not_training_approval(self):
        n=copy.deepcopy(self.n);n['decisions'][0][1]='content_observed'
        f=summarize_frames(self.r,n,validate_observations(self.r,n,'hash'))[0]
        self.assertFalse(f['training_ready']);self.assertEqual(f['status'],'visual_content_reviewed_only')

    def test_unknown_unboxed_content_blocks(self):
        n=copy.deepcopy(self.n);n['decisions'][0][1]='content_observed'
        n['frame_checks']['N01']='unboxed_blue_identity_pending'
        f=summarize_frames(self.r,n,validate_observations(self.r,n,'hash'))[0]
        self.assertEqual(f['status'],'held_for_evidence')

    def test_automatic_pass_status_rejected(self):
        n=copy.deepcopy(self.n);n['decisions'][0][1]='accepted'
        with self.assertRaises(ValueError):validate_observations(self.r,n,'hash')
