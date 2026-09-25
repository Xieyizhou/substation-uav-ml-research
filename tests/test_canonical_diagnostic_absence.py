import unittest
from src.vision.canonical.collect import allows_diagnostic_absence, valid_annotation_mode, violates_no_target_gate


class DiagnosticAbsenceTests(unittest.TestCase):
    def test_only_explicit_nontraining_calibration(self):
        plan=dict(diagnostic_only=True, training_admitted=False,
                  diagnostic_allow_expected_absence=True)
        self.assertTrue(allows_diagnostic_absence(plan, 'calibration'))
        self.assertFalse(allows_diagnostic_absence({**plan,'diagnostic_require_expected_presence':True}, 'calibration'))
        self.assertFalse(allows_diagnostic_absence(plan, 'pilot'))
        for key in plan:
            incomplete={k:v for k,v in plan.items() if k!=key}
            self.assertFalse(allows_diagnostic_absence(incomplete, 'calibration'))
        self.assertFalse(allows_diagnostic_absence({**plan,'training_admitted':True}, 'calibration'))

    def test_full_2d_requires_explicit_visual_instance_labels(self):
        self.assertFalse(valid_annotation_mode({'annotation_mode':'full_2d','label_mode':'source'}))
        self.assertFalse(valid_annotation_mode({'annotation_mode':'full_2d','label_mode':'visual-instance'}))
        self.assertTrue(valid_annotation_mode({'annotation_mode':'full_2d','label_mode':'visual-instance','hierarchy_mode':'top-level-equipment'}))
        self.assertTrue(valid_annotation_mode({'annotation_mode':'visible_2d','label_mode':'source'}))

    def test_negative_frame_gate_rejects_any_target_class(self):
        plan={'diagnostic_require_no_targets':True}
        self.assertTrue(violates_no_target_gate(plan,{'cabinet','switchgear'}))
        self.assertFalse(violates_no_target_gate(plan,{'cabinet','control_building'}))
        self.assertFalse(violates_no_target_gate({}, {'switchgear'}))
