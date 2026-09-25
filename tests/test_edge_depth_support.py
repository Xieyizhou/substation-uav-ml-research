import numpy as np
import unittest
from scripts.vision.audit_edge_depth_support import ray_box


def test_ray_box_parallel_and_miss():
    rays = np.array([[1., 0, 0], [0, 1., 0], [-1., 0, 0]])
    result = ray_box(np.zeros(3), rays, np.array([2., -1, -1]), np.array([3., 1, 1]))
    assert result[0] == 2
    assert np.isinf(result[1:]).all()


def test_ray_box_inside_and_tangent():
    assert ray_box(np.zeros(3), np.array([1., 0, 0]), -np.ones(3), np.ones(3)) == 0
    assert ray_box(np.array([0., 1, 0]), np.array([1., 0, 0]),
                   np.array([2., -1, -1]), np.array([3., 1, 1])) == 2


class DepthGeometryTests(unittest.TestCase):
    def test_parallel_and_miss(self):
        test_ray_box_parallel_and_miss()

    def test_inside_and_tangent(self):
        test_ray_box_inside_and_tangent()
