import math
from types import SimpleNamespace
import unittest
from src.flight.live_obstacle_map import LiveObstacleMap,GazeboScanGate
from src.planner.local_frame import LocalFrame
from src.flight.replanning_controller import attempt_local_replan,empty_replan_state


class LiveObstacleMapTests(unittest.TestCase):
    def test_no_return_is_distinct_from_invalid_or_frozen_sensor(self):
        scan=SimpleNamespace(sequence=1,timestamp_s=1.,angle_min_rad=-2.356195,angle_max_rad=2.356195,angle_step_rad=4.71239/1079,range_min_m=.1,range_max_m=30.,ranges_m=[math.inf]*1080,age_s=lambda now:0.)
        source=SimpleNamespace(latest=lambda:scan,health=lambda:SimpleNamespace(healthy=True))
        gate=GazeboScanGate();self.assertEqual(gate.nearest(source),30.)
        scan.sequence=2
        with self.assertRaises(RuntimeError):gate.nearest(source)
        scan.timestamp_s=2.;self.assertEqual(gate.nearest(source),30.)
        for bad in (math.nan,-math.inf,0.,31.):
            scan.ranges_m[0]=bad
            with self.assertRaises(RuntimeError):gate.nearest(source)
        scan.ranges_m[0]=1.4;self.assertEqual(gate.nearest(source),1.4)
        scan.age_s=lambda now:1.
        with self.assertRaises(RuntimeError):gate.nearest(source)

    def scan(self):
        return SimpleNamespace(sequence=1,age_s=lambda now:0.,ranges_m=[2.2]*5,range_min_m=.1,range_max_m=30.,angle_at=lambda i:(i-2)*.02)

    def test_observed_evidence_changes_route_without_fixture_identity(self):
        m=LiveObstacleMap([],LocalFrame(1.3,1.5));s=self.scan()
        self.assertFalse(m.route_blocked([(1.3,1.5),(1.3,6)]))
        self.assertEqual(m.ingest(s,dict(north=0.,east=0.),dict(roll=0.,pitch=0.,yaw=0.),0),5)
        self.assertTrue(m.route_blocked([(1.3,1.5),(1.3,6)]))
        route=m.replan((1.3,1.5),(1.3,6))
        self.assertFalse(m.route_blocked(route));self.assertGreater(max(p[0] for p in route),3.)
        self.assertEqual(m.ingest(s,dict(north=0.,east=0.),dict(roll=0.,pitch=0.,yaw=0.),0),0)

    def test_stale_tilt_and_blocked_start_rejected(self):
        m=LiveObstacleMap([],LocalFrame());s=self.scan();s.age_s=lambda now:1.
        with self.assertRaises(ValueError):m.ingest(s,{}, {},1.)
        s=self.scan()
        with self.assertRaises(ValueError):m.ingest(s,{},dict(roll=2.,pitch=0.),0)
        m.points=[(1.,1.)]*3
        with self.assertRaises(ValueError):m.replan((1.,1.),(4.,4.))

    def test_legacy_replan_must_not_clear_blocked_goal(self):
        config=dict(goal_cell=(3,3),static_obstacles={(3,3)},dynamic_obstacle_inflation=0,width=5,height=5,allow_diagonal=False,resolution_m=1.)
        result=attempt_local_replan(config,empty_replan_state(),SimpleNamespace(east_m=.5,north_m=.5),{},1.)
        self.assertIsNone(result)
