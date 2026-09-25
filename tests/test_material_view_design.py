import unittest
from scripts.vision.verify_material_view_design import blockers

class DesignTests(unittest.TestCase):
    def row(self,**kw):
        return dict(status='reviewed_candidate_only',held_labels=[],unboxed_visible_instances=[],lineage_id='seen-pose',source_pose_already_exposed=True,altered_target_class='switchgear',**kw)
    def test_variants_not_new_poses(self):
        r=blockers([self.row(),self.row()]);self.assertEqual(r['new_pose_groups'],0);self.assertEqual(r['usable_pose_groups'],1)
    def test_in_frame_class_not_material_coverage(self):
        r=blockers([self.row(full_label_classes={'reactor':1,'switchgear':3})]);self.assertIn('reactor',r['missing_altered_classes'])
    def test_held_not_usable(self):
        x=self.row();x['held_labels']=['uncertain'];self.assertEqual(blockers([x])['usable_variants'],0)
    def test_missing_label_not_usable(self):
        x=self.row();x['unboxed_visible_instances']=['missing'];self.assertEqual(blockers([x])['usable_variants'],0)

if __name__=='__main__':unittest.main()
