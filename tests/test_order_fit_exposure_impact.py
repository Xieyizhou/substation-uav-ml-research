import unittest
from scripts.vision.order_fit_exposure_impact import subset_impact
from scripts.vision.order_fit_legacy_pose_exclusion import distance


class ExposureImpactTests(unittest.TestCase):
    def test_quaternion_sign_equivalence(self):
        a={'position':[0,0,0],'orientation':[0,0,0,1]}
        b={'position':[0,0,0],'orientation':[0,0,0,-1]}
        self.assertEqual(distance(a,b),(0,0))

    def test_invalid_pose_rejected(self):
        a={'position':[0,0,0],'orientation':[0,0,0,0]}
        with self.assertRaises(ValueError):distance(a,a)

    def test_removed_draws_and_capacity(self):
        m=[dict(member_id='a',class_instances={'x':5}),dict(member_id='b',class_instances={'x':1}),dict(member_id='c',class_instances={})]
        r=subset_impact(m,{'a':2,'b':7},{'a','c'})
        self.assertEqual(r['draws_requiring_reassignment'],7)
        self.assertEqual(r['minimum_possible_max_draws_using_all_remaining'],5)
        self.assertEqual(r['minimum_possible_max_draws_without_restoring_zero_exposure'],9)
        self.assertFalse(r['same_subset_class_necessary_conditions'][0]['same_subset_exact_preservation_passes_necessary_divisibility'])

    def test_no_remaining_members(self):
        r=subset_impact([dict(member_id='a',class_instances={'x':1})],{'a':3},set())
        self.assertIsNone(r['minimum_possible_max_draws_using_all_remaining'])
        self.assertFalse(r['same_subset_class_necessary_conditions'][0]['same_subset_exact_preservation_passes_necessary_divisibility'])

    def test_zero_class_exposure(self):
        r=subset_impact([dict(member_id='a',class_instances={'x':1})],{},set())
        self.assertTrue(r['same_subset_class_necessary_conditions'][0]['same_subset_exact_preservation_passes_necessary_divisibility'])


if __name__=='__main__':unittest.main()
