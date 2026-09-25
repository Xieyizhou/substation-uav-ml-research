import unittest
import numpy as np
from scripts.vision.validate_edge_raster import clip_polygon,coverage,bbox,scanline_extent
from scripts.vision.finalize_edge_raster_validation import compare_pose,POSES,read


class RasterTests(unittest.TestCase):
    def test_pixel_center(self):
        p=np.array([[0,2],[2,0],[2,2]],float)
        result=coverage([p],2,2)
        self.assertEqual(bbox(result),[0,0,2,2])
        self.assertEqual(scanline_extent([p],1.5),[.5,2])

    def test_empty(self):
        self.assertIsNone(bbox(coverage([],2,2)))
        self.assertEqual(clip_polygon([]),[])

    def test_nonfinite_reject(self):
        with self.assertRaises(ValueError):clip_polygon([[0,0,float('nan'),1]])

    def test_behind_camera(self):
        self.assertEqual(clip_polygon([[0,0,0,-1],[.1,0,0,-1],[0,.1,0,-1]]),[])

    def receipts(self):
        return [read(POSES/v/'replay/T023/attempt-01/receipt.json') for v in ('control','clipped')]

    def test_actual_regression_held(self):
        r=compare_pose(*self.receipts())
        self.assertEqual(r['status'],'held_existing_box_change')
        self.assertGreater(r['stable_comparisons'][0]['over_one_pixel']['139'],36)

    def test_missing_record_rejected(self):
        a,b=self.receipts();b['records'].pop()
        with self.assertRaises(ValueError):compare_pose(a,b)

    def test_rgb_mismatch_rejected(self):
        a,b=self.receipts();b['records'][0]['rgb_exact']=False
        with self.assertRaises(ValueError):compare_pose(a,b)

    def test_mask_change_rejected(self):
        a,b=self.receipts();b['records'][0]['mask_sha256']='changed'
        with self.assertRaises(ValueError):compare_pose(a,b)


if __name__=='__main__':unittest.main()
