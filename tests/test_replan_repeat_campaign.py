import json
from pathlib import Path
import tempfile
import unittest

from scripts.flight.replan_repeat_fixture import CASES, ready
from scripts.flight.replan_repeat_campaign import validate_hashes
from scripts.flight.prepare_avoidance_route import build, collision_boxes
from src.ml.artifacts import file_sha256


class RepeatCampaignTests(unittest.TestCase):
    def test_trigger_requires_actual_northward_motion_and_distance(self):
        self.assertFalse(ready('baseline',(1.2,.9),dict(vn=.149)))
        self.assertTrue(ready('baseline',(1.2,.9),dict(vn=.15)))
        self.assertFalse(ready('later',(1.2,1.04),dict(vn=.2)))
        self.assertTrue(ready('later',(1.2,1.05),dict(vn=.2)))

    def test_geometry_has_positive_and_negative_controls(self):
        boxes=collision_boxes(Path('simulation/worlds/substation_simple.sdf'))
        for name,c in CASES.items():
            b=dict(x0=c['east']-.3,x1=c['east']+.3,y0=c['north']-.3,y1=c['north']+.3,z0=0,z1=4)
            if c['expected']=='goal':
                self.assertTrue(build(boxes+[b],start=(1.2,1.3),goal=(1.3,6.5)))
            else:
                with self.assertRaisesRegex(ValueError,r'No A\* path found'):
                    build(boxes+[b],start=(1.2,1.3),goal=(1.3,6.5))

    def test_changed_or_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json';p.write_text(json.dumps({'version':1}))
            r=dict(inputs={str(p):file_sha256(p)});validate_hashes(r)
            p.write_text(json.dumps({'version':2}))
            with self.assertRaises(ValueError):validate_hashes(r)
            p.unlink()
            with self.assertRaises(OSError):validate_hashes(r)
