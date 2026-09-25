import unittest
from scripts.vision.audit_revision_exposure import impact


class ImpactTests(unittest.TestCase):
    def test_instance_multiplicity_and_zero(self):
        d=['a']*2700
        r=impact(d,d,[dict(member_id='a',class_instances={'switchgear':3}),dict(member_id='b',class_instances={'reactor':1})])
        self.assertEqual(r['whole_frame_hold_counterfactual']['existing_label_instance_exposures']['switchgear'],8100)
        self.assertEqual(r['members'][1]['actual_image_exposures'],0)
        self.assertEqual(sum(r['members'][0]['windows_50_steps']),2700)
        self.assertFalse(r['whole_frame_hold_counterfactual']['applied'])

    def test_order_mismatch(self):
        with self.assertRaises(ValueError):impact(['a']*2700,['b']*2700,[])

    def test_short_sequence(self):
        with self.assertRaises(ValueError):impact(['a'],['a'],[])
