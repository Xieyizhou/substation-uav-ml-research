import unittest

from src.vision.canonical.occlusion import segment_intersects_bounds, view_blockers


class OcclusionTests(unittest.TestCase):
    def test_crossing_and_parallel(self):
        box = [1, 2, -1, 1, -1, 1]
        self.assertTrue(segment_intersects_bounds([0, 0, 0], [3, 0, 0], box))
        self.assertFalse(segment_intersects_bounds([0, 2, 0], [3, 2, 0], box))
        self.assertFalse(segment_intersects_bounds([0, 0, 0], [.5, 0, 0], box))

    def test_tangent_and_inside(self):
        box = [1, 2, -1, 1, -1, 1]
        self.assertTrue(segment_intersects_bounds([0, 1, 0], [3, 1, 0], box))
        self.assertTrue(segment_intersects_bounds([1.5, 0, 0], [1.5, 0, 0], box))

    def test_target_excluded_and_height(self):
        objects = [{'name': 'target', 'bounds': [4, 6, -1, 1, 0, 2]},
                   {'name': 'wall', 'bounds': [2, 3, -1, 1, 0, 2]}]
        self.assertEqual(view_blockers({'object_id': 'target',
                                       'camera_position': [0, 0, 1]}, objects), ['wall'])
        self.assertEqual(view_blockers({'object_id': 'target',
                                       'camera_position': [0, 0, 10]}, objects), [])

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            segment_intersects_bounds([float('nan'), 0, 0], [0, 0, 0], [0, 1] * 3)
        with self.assertRaises(ValueError):
            view_blockers({'object_id': 'missing', 'camera_position': [0, 0, 0]}, [])
