import unittest
from scripts.vision.correct_material_trajectory_labels import classify

class CorrectLabels(unittest.TestCase):
    def test_learning_is_not_loss(self):
        self.assertEqual(classify([False,False,True,True]),'endpoint_hit')
        self.assertEqual(classify([False,True,False,True]),'endpoint_hit_with_intermediate_loss')
        self.assertEqual(classify([True,False,True]),'endpoint_hit_with_intermediate_loss')
        self.assertEqual(classify([False,True,False]),'previously_hit_endpoint_miss')
        self.assertEqual(classify([False,False]),'never_hit_at_observed_steps')

if __name__=='__main__':unittest.main()
