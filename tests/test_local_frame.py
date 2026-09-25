import math
from types import SimpleNamespace
import unittest

from src.planner.local_frame import LocalFrame
from src.planner.astar_grid import cell_to_local_waypoint
from src.flight.route_planning import plan_path
from src.flight.replanning_controller import local_position_to_grid_cell, replanned_path_to_waypoints
from src.perception.lidar_detector import LidarRiskDetector


class LocalFrameTests(unittest.TestCase):
    def test_spawn_offset_reproduction_and_fix(self):
        self.assertEqual(cell_to_local_waypoint((1,1))['east_m'],1.5)
        frame=dict(east_offset_m=1.5,north_offset_m=1.5)
        w=cell_to_local_waypoint((1,1),local_frame=frame)
        self.assertEqual((w['east_m'],w['north_m']),(0.,0.))
        self.assertEqual(local_position_to_grid_cell(SimpleNamespace(east_m=0.,north_m=0.),1.,frame),(1,1))

    def test_rotation_round_trip(self):
        for angle in (0,25,90,-90,180):
            f=LocalFrame(1.5,-2.7,angle)
            for e,n in ((0,0),(2,-3),(-1,8)):
                a,b=f.to_local(*f.to_map(e,n))
                self.assertAlmostEqual(a,e);self.assertAlmostEqual(b,n)

    def test_planner_replan_and_lidar_share_frame(self):
        frame=dict(east_offset_m=1.5,north_offset_m=1.5,rotation_deg=0.)
        config=dict(start=(1,1),goal=(4,1),obstacles=set(),width=8,height=8,resolution_m=1.,altitude_m=1.,local_frame=frame)
        path,_,waypoints=plan_path(config,False)
        replan=replanned_path_to_waypoints(path,config)
        self.assertEqual(waypoints[-1]['east_m'],3.)
        self.assertEqual(replan[-1]['east_m'],3.)
        detector=LidarRiskDetector(None,resolution_m=1.,local_frame=frame)
        x,y,n,e=detector._global_cell(3.,0.,0.,0.,90.)
        self.assertEqual((x,y),(4,1))
        self.assertAlmostEqual(e,3.)
        self.assertEqual(local_position_to_grid_cell(SimpleNamespace(east_m=e,north_m=n),1.,frame),(4,1))

    def test_invalid_and_negative_coordinates(self):
        with self.assertRaises(ValueError):LocalFrame(float('nan'))
        with self.assertRaises(ValueError):LocalFrame().cell(0,0,0)
        with self.assertRaises(ValueError):LocalFrame().to_map(math.inf,0)
        self.assertEqual(LocalFrame().cell(-.1,-.1,1.),(-1,-1))
