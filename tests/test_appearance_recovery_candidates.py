import unittest
from scripts.vision.prepare_appearance_recovery_candidates import near

class PoseNoveltyTests(unittest.TestCase):
    def test_same_and_quaternion_sign(self):
        a=dict(position=[0,0,0],orientation=[0,0,0,1])
        self.assertTrue(near(a,a))
        self.assertTrue(near(a,{**a,'orientation':[0,0,0,-1]}))
        self.assertFalse(near(a,{**a,'position':[.5,0,0]}))
        self.assertFalse(near(a,{**a,'orientation':[0,0,1,0]}))
    def test_zero_orientation_rejected(self):
        with self.assertRaises(ValueError):near(dict(position=[0,0,0],orientation=[0]*4),dict(position=[0,0,0],orientation=[0]*4))

if __name__=='__main__':unittest.main()
