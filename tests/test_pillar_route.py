import copy
import unittest
from scripts.flight.fly_pillar_route import cruise_guard,segment_distance,angle_error
from scripts.flight import route_trial_lifecycle as trial
from src.planner.local_frame import LocalFrame
from scripts.vision.material_shadow_route import validate_duration


class PillarRouteTests(unittest.TestCase):
    def test_route_sidecar_duration_contract(self):
        validate_duration(240)
        for v in (0,-1,240.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):validate_duration(v)

    def state(self):
        return dict(local=(10.,dict(east=0.,north=2.,down=-2.,vn=.2,ve=0.,vd=0.)),attitude=(10.,dict(roll=0.,pitch=0.,yaw=0.)))

    def test_distance_and_yaw_wrap(self):
        self.assertAlmostEqual(segment_distance((.1,2),(0,0),(0,3)),.1)
        self.assertEqual(segment_distance((0,4),(0,0),(0,3)),1.)
        self.assertEqual(angle_error(-179,179),2.)

    def test_cruise_gates(self):
        origin=dict(east=0.,north=0.,down=0.)
        state=self.state()
        cruise_guard(state,origin,LocalFrame(),(0,0),(0,3),10.1)
        for field,value in [('east',.181),('down',-1.74),('down',-2.26),('vn',.351),('east',float('nan'))]:
            s=copy.deepcopy(state);s['local'][1][field]=value
            with self.assertRaises(RuntimeError):cruise_guard(s,origin,LocalFrame(),(0,0),(0,3),10.1)
        with self.assertRaises(RuntimeError):cruise_guard(state,origin,LocalFrame(),(0,0),(0,3),11.1)
        s=copy.deepcopy(state);s['attitude'][1]['pitch']=5.01
        with self.assertRaises(RuntimeError):cruise_guard(s,origin,LocalFrame(),(0,0),(0,3),10.1)

    def test_default_envelope_is_not_widened(self):
        with self.assertRaises(RuntimeError):trial.guard(self.state(),10.1,dict(east=0.,north=0.,down=0.),max_height_m=2.8)
        trial.guard(self.state(),10.1,dict(east=0.,north=0.,down=0.),max_height_m=2.8,max_drift_m=6.)
