import unittest
import numpy as np
from scripts.vision.diagnose_full_box_projection import SIGNS, legacy_reject, outside_same_side, near_clip_box, project, translation, ET, reconstruct, read, PRIOR


class ProjectionTests(unittest.TestCase):
    def test_crossing_interval_false_rejection(self):
        self.assertTrue(legacy_reject([-2, -.2], [2, .2]))
        self.assertFalse(outside_same_side([-2, -.2], [2, .2]))

    def test_same_side_and_inside_controls(self):
        for low, high, expected in [([2,0],[3,.2],True), ([-3,0],[-2,.2],True), ([-.5,-.5],[.5,.5],False)]:
            self.assertEqual(legacy_reject(low,high),expected)
            self.assertEqual(outside_same_side(low,high),expected)

    def test_near_clipping_before_division(self):
        box=np.array(SIGNS,dtype=float)
        clipped=near_clip_box(box,.1)
        self.assertEqual(len(clipped),8)
        self.assertTrue(np.all(clipped[:,0]>=.1-1e-12))
        self.assertTrue(np.isfinite(project(clipped,1.466,1080/1920)).all())

    def test_behind_camera_empty(self):
        box=np.array(SIGNS,dtype=float);box[:,0]-=3
        self.assertEqual(len(near_clip_box(box,.1)),0)
        with self.assertRaises(ValueError):project(np.zeros((1,3)),1.466,.5625)

    def test_projection_axes(self):
        np.testing.assert_allclose(project(np.array([[1,0,0],[1,-1,0],[1,0,1]]),np.pi/2,1),[[0,0],[1,0],[0,1]],atol=1e-12)

    def test_unsupported_transform_rejected(self):
        for xml in ['<visual><pose relative_to="a">0 0 0 0 0 0</pose></visual>', '<visual><pose>0 0 0 0 0 1</pose></visual>']:
            with self.assertRaises(ValueError):translation(ET.fromstring(xml))

    def test_frozen_t027_analytical_reproduction(self):
        result=reconstruct(read(PRIOR/'protocol.json')['frames'][0])
        by_name={v['visual']:v for v in result['visuals']}
        self.assertEqual(set(by_name),{'base','body','front_panel'})
        for v in by_name.values():
            self.assertTrue(v['legacy_rejected'])
            self.assertLess(v['depth_range'][0],0)
            self.assertGreater(v['depth_range'][1],0)
        self.assertEqual([v['near_clipped_extent_intersects_viewport'] for v in result['visuals']],[False,True,False])
        predicted_top=(1-by_name['body']['near_clipped_ndc_bounds'][1][1])*1080/2
        self.assertLess(abs(predicted_top-1052),1)


if __name__=='__main__':unittest.main()
