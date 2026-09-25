import unittest
from scripts.vision.design_full_scene_poses import projected_bounds,screen

class FullScenePoseTests(unittest.TestCase):
    def setUp(self):
        self.v={'camera_position':[0,0,0],'orientation':[0,0,0,1],'object_id':'target'}
    def test_complete_box_inside(self):
        b=projected_bounds([5,6,-.5,.5,-.5,.5],self.v)
        self.assertGreater(b[0],16);self.assertLess(b[2],1904)
    def test_behind_camera(self):self.assertIsNone(projected_bounds([-6,-5,-1,1,-1,1],self.v))
    def test_edge_crossing_rejected(self):
        with self.assertRaises(ValueError):projected_bounds([5,6,3,7,-.5,.5],self.v)
    def test_near_crossing_rejected(self):
        with self.assertRaises(ValueError):projected_bounds([.05,.2,-.01,.01,-.01,.01],self.v)
    def test_nonplanned_edge_also_rejected(self):
        objects=[{'name':'target','category':'switchgear','bounds':[5,6,-.5,.5,-.5,.5]},
                 {'name':'other','category':'transformer','bounds':[5,6,3,7,-.5,.5]}]
        with self.assertRaises(ValueError):screen(self.v,objects)
    def test_foreground_blocker(self):
        objects=[{'name':'target','category':'switchgear','bounds':[5,6,-.5,.5,-.5,.5]},
                 {'name':'building','category':'building','bounds':[2,3,-1,1,-1,1]}]
        with self.assertRaises(ValueError):screen(self.v,objects)
