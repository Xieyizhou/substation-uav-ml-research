import json
import unittest
from scripts.flight.fly_px4_shadow_hover import check_spawn, guard, ROOT


class HoverTrialTests(unittest.TestCase):
    def test_spawn_envelope(self):
        config=json.loads((ROOT/'config/substation_obstacles.json').read_text())
        check_spawn(config)
        with self.assertRaises(ValueError):check_spawn(config,x=-10)
        b=config['obstacles'][0];ox,oy,_=config['gazebo_world_origin_m']
        with self.assertRaises(ValueError):check_spawn(config,x=ox+b['x_min'],y=oy+b['y_min'])

    def test_guards(self):
        origin=dict(north=0.,east=0.,down=0.)
        def state():return dict(local=(10.,dict(north=0.,east=0.,down=-1.,vd=0.)),attitude=(10.,dict(roll=0.,pitch=0.,yaw=0.)))
        self.assertEqual(guard(state(),10.,origin),(1.,0.))
        with self.assertRaises(RuntimeError):guard(state(),12.,origin)
        for field,value in [('north',1.1),('down',-1.9),('down',float('nan'))]:
            s=state();s['local'][1][field]=value
            with self.assertRaises(RuntimeError):guard(s,10.,origin)
        s=state();s['attitude'][1]['roll']=21.
        with self.assertRaises(RuntimeError):guard(s,10.,origin)
