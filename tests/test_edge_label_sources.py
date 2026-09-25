import unittest
from scripts.vision.audit_edge_label_sources import hull,clipped_area

class EdgeProjectionTests(unittest.TestCase):
    def test_hull_and_full_rectangle(self):
        polygon=hull([(0,0),(10,0),(10,10),(0,10),(5,5),(0,0)])
        self.assertEqual(len(polygon),4);self.assertEqual(clipped_area(polygon,10,10),100)
    def test_clip_partial_and_outside(self):
        self.assertEqual(clipped_area(hull([(-10,0),(5,0),(5,10),(-10,10)]),10,10),50)
        self.assertEqual(clipped_area(hull([(-10,0),(-5,0),(-5,10),(-10,10)]),10,10),0)
    def test_bbox_does_not_establish_occupancy(self):
        # The bounding rectangle intersects the image, but most of it is empty.
        poly=hull([(-100,0),(1,99),(-100,100)])
        self.assertLess(clipped_area(poly,100,100),1)

if __name__=='__main__':unittest.main()
