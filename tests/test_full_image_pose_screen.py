import unittest
from scripts.vision.screen_full_image_poses import screen

class FullScreenTests(unittest.TestCase):
    def test_target_clear_other_truncated(self):
        v=dict(orientation=[0,0,0,1],camera_position=[0,0,1],object_id='target')
        target=dict(name='target',category='reactor',bounds=[9,10,-1,1,0,2])
        self.assertFalse(screen(v,[target])['risks'])
        other=dict(name='other',category='switchgear',bounds=[2,4,2,5,0,2])
        self.assertIn(('other','projected_bounds_truncated'),screen(v,[target,other])['risks'])
    def test_occluder(self):
        v=dict(orientation=[0,0,0,1],camera_position=[0,0,1],object_id='target')
        objs=[dict(name='target',category='reactor',bounds=[9,10,-1,1,0,2]),dict(name='wall',category='wall',bounds=[5,6,-2,2,0,3])]
        self.assertIn(('target','center_ray_blocked'),screen(v,objs)['risks'])

if __name__=='__main__':unittest.main()
