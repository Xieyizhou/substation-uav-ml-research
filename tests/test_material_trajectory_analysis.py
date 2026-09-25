import unittest
from scripts.vision.analyze_material_learning_trajectory import classify


class TrajectoryClassification(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify([False,False,False]),'never_hit_at_observed_steps')
        self.assertEqual(classify([False,True,False]),'previously_hit_endpoint_miss')
        self.assertEqual(classify([True,False,False]),'previously_hit_endpoint_miss')
        self.assertEqual(classify([False,False,True]),'endpoint_hit')
        self.assertEqual(classify([True,False,True]),'endpoint_hit_with_intermediate_loss')
        with self.assertRaises(ValueError):classify([])


if __name__=='__main__':unittest.main()
