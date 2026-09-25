import json
import math
from pathlib import Path
import tempfile
import unittest
from src.flight.clearance_route import clearance_route
from src.flight.replan_evidence import EvidenceJournal,scan_record
from src.planner.local_frame import LocalFrame
from src.sensors.types import LaserScanFrame
from scripts.flight.prepare_avoidance_route import inside


class ClearanceRouteTests(unittest.TestCase):
    def test_hard_boundary_and_determinism(self):
        b=dict(x0=3,x1=3.3,y0=3,y1=3.3)
        path=clearance_route([b],(1,1),(6,6))
        self.assertEqual(path,clearance_route([b],(1,1),(6,6)))
        for a,c in zip(path,path[1:]):
            n=max(1,math.ceil(math.dist(a,c)/.01))
            for i in range(n+1):
                self.assertFalse(inside(tuple(x+(y-x)*i/n for x,y in zip(a,c)),b,1.8))
        for start,end in [((3,3),(6,6)),((1,1),(3,3))]:
            with self.assertRaises(ValueError):clearance_route([b],start,end)

    def test_no_path_stays_blocked(self):
        b=dict(x0=0,x1=20,y0=3,y1=4)
        with self.assertRaisesRegex(ValueError,'No A'):
            clearance_route([b],(1,1),(1,8))

    def test_raw_scan_preserves_infinity_and_repeated_decisions(self):
        scan=LaserScanFrame(1,1,'lidar',-1,1,1,.1,30,(float('inf'),2.,3.),'test',1)
        self.assertEqual(LaserScanFrame.from_record(scan_record(scan)).ranges_m,scan.ranges_m)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);j=EvidenceJournal(p,LocalFrame());state=dict(local=(1,{'north':0}),attitude=(1,{'yaw':0}))
            j.append(scan,state,1,0,False,0);j.append(scan,state,1,0,True,3);j.close()
            rows=[json.loads(v) for v in (p/'scan-pose.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertEqual(rows[1]['new_returns'],3)
