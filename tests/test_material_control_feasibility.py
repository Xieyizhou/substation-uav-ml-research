import unittest
from scripts.vision.material_control_feasibility import resolve_target, check_candidate_frames

class IdentityTests(unittest.TestCase):
    def test_nonplanned_target_uses_mapping(self):
        t={'annotation_id':'source-instance-0128-box-0','class_name':'switchgear'}
        self.assertEqual(resolve_target(t,{128:dict(object_id='west_switchgear_02',category='switchgear')})['object_id'],'west_switchgear_02')

    def test_unknown_and_colliding_identity_fail(self):
        t={'annotation_id':'source-instance-0128-box-0','class_name':'switchgear'}
        for m in ({}, {128:dict(object_id='same',category='switchgear'),139:dict(object_id='same',category='switchgear')},
                  {128:dict(object_id='x',category='transformer')}):
            with self.assertRaises(ValueError): resolve_target(t,m)

    def test_good_existing_labels_do_not_clear_extra_instance(self):
        row=dict(source_review_id='T027',variant='warm',status='reviewed_candidate_only',held_labels=[],unboxed_visible_instances=[128])
        self.assertEqual(len(check_candidate_frames([row])),1)

    def test_candidate_not_training_admission(self):
        row=dict(source_review_id='T020',variant='warm',status='reviewed_candidate_only',held_labels=[],unboxed_visible_instances=[])
        self.assertEqual(check_candidate_frames([row]),[])
        with self.assertRaises(ValueError): check_candidate_frames([row,row])

    def test_held_content_blocks(self):
        row=dict(source_review_id='T036',variant='cool',status='held_whole_image',held_labels=['03'],unboxed_visible_instances=[])
        self.assertEqual(len(check_candidate_frames([row])),1)
