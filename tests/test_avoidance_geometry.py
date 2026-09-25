import math
import unittest
from src.planner.astar_grid import astar
from src.perception.lidar_detector import LidarRiskDetector
from scripts.flight.prepare_avoidance_route import build,collision_boxes,ROOT


class AvoidanceGeometryTests(unittest.TestCase):
    def test_diagonal_cannot_escape_blocked_corner(self):
        with self.assertRaisesRegex(ValueError,'No A\\* path'):
            astar((0,0),(1,1),{(1,0),(0,1)},2,2,True)

    def test_single_blocked_side_forces_detour(self):
        p=astar((0,0),(1,1),{(1,0)},2,2,True)
        self.assertEqual(p,[(0,0),(0,1),(1,1)])
        self.assertEqual(astar((0,0),(1,1),set(),2,2,True),[(0,0),(1,1)])

    def test_mount_translation_rotates_with_compass_yaw(self):
        d=LidarRiskDetector(None,resolution_m=.1,sensor_forward_m=-.1,sensor_left_m=.2)
        for yaw,expected in ((0,(1.9,-.2)),(90,(.2,1.9)),(180,(-1.9,.2)),(270,(-.2,-1.9))):
            _,_,n,e=d._global_cell(2,0,0,0,yaw)
            self.assertAlmostEqual(n,expected[0]);self.assertAlmostEqual(e,expected[1])

    def test_nonfinite_mount_rejected(self):
        with self.assertRaises(ValueError):LidarRiskDetector(None,resolution_m=1,sensor_forward_m=math.nan)

    def test_collision_inventory_uses_world_not_grid_rounding(self):
        boxes=collision_boxes(ROOT/'simulation/worlds/substation_simple.sdf')
        self.assertEqual(len(boxes),14)
        transformer=next(b for b in boxes if '/transformer_1/' in b['name'])
        self.assertAlmostEqual(transformer['x0'],5.3)
        self.assertAlmostEqual(transformer['z1'],1.9)

    def test_real_scene_blocks_conservative_wall_route(self):
        boxes=collision_boxes(ROOT/'simulation/worlds/substation_simple.sdf')
        wall=dict(name='wall',x0=3.9,x1=4.1,y0=1.,y1=2.,z0=0.,z1=4.)
        with self.assertRaisesRegex(ValueError,'No A\\* path'):build(boxes+[wall])
        points=build([wall])
        self.assertEqual(points[0],(1.5,1.5))
        self.assertGreater(max(n for e,n in points),3.8)
