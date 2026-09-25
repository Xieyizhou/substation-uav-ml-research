import unittest
from scripts.vision.screen_full_image_poses_v2 import screen

class FrustumScreenTests(unittest.TestCase):
    def test_offscreen_near_plane_object_does_not_reject(self):
        view=dict(orientation=[0,0,0,1],camera_position=[0,0,1],object_id='target')
        target=dict(name='target',category='reactor',bounds=[9,10,-1,1,0,2])
        side=dict(name='side',category='switchgear',bounds=[-1,1,10,11,0,2])
        self.assertEqual(screen(view,[target,side])['risks'],[])
    def test_visible_fringe_still_rejected(self):
        view=dict(orientation=[0,0,0,1],camera_position=[0,0,1],object_id='target')
        target=dict(name='target',category='reactor',bounds=[9,10,-1,1,0,2])
        fringe=dict(name='fringe',category='switchgear',bounds=[2,4,2,5,0,2])
        self.assertIn(('fringe','projected_bounds_truncated'),screen(view,[target,fringe])['risks'])

if __name__=='__main__':unittest.main()
