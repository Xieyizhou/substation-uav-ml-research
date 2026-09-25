import unittest
from scripts.vision.build_recovery_control_matrix import region


class ProjectedRegionTests(unittest.TestCase):
    def test_centered_box_and_distance(self):
        near=region([5,6,-1,1,-1,1],[0,0,0],[0,0,0,1])
        far=region([10,11,-1,1,-1,1],[0,0,0],[0,0,0,1])
        self.assertAlmostEqual((near[0]+near[2])/2,960)
        self.assertAlmostEqual((near[1]+near[3])/2,540)
        self.assertGreater(near[2]-near[0],far[2]-far[0])

    def test_behind_camera_rejected(self):
        self.assertIsNone(region([-2,-1,-1,1,-1,1],[0,0,0],[0,0,0,1]))
