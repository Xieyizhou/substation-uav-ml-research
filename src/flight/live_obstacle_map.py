"""Sensor-only additions to a frozen map for bounded static-obstacle trials.

Returns are retained; lack of a return never clears previously observed
occupancy. This is not moving-object tracking or unknown-volume certification.
"""
import math
import time
from src.flight.sitl_support import inside,build
from src.perception.lidar_detector import LidarRiskDetector


class GazeboScanGate:
    """Specific frozen research scanner; +inf is outside detection range."""
    def __init__(self):self.last_sequence=None;self.last_timestamp=None

    def nearest(self,source):
        s=source.latest()
        if s is None or not source.health().healthy or s.age_s(time.monotonic())>.5:raise RuntimeError('Missing/stale scan')
        if len(s.ranges_m)!=1080 or abs(s.range_min_m-.1)>1e-5 or abs(s.range_max_m-30.)>1e-5:raise RuntimeError('Unexpected scanner contract')
        if not all(math.isfinite(v) for v in (s.timestamp_s,s.angle_min_rad,s.angle_max_rad,s.angle_step_rad,s.range_min_m,s.range_max_m)):raise RuntimeError('Nonfinite scan metadata')
        if abs(s.angle_min_rad+2.356195)>1e-5 or abs(s.angle_max_rad-2.356195)>1e-5 or abs(s.angle_step_rad*1079-4.71239)>1e-4:raise RuntimeError('Unexpected scan geometry')
        if s.sequence!=self.last_sequence:
            if self.last_timestamp is not None and s.timestamp_s<=self.last_timestamp:raise RuntimeError('Scan capture timestamp not advancing')
            self.last_sequence=s.sequence;self.last_timestamp=s.timestamp_s
        if any(math.isnan(v) or v<s.range_min_m or (math.isfinite(v) and v>s.range_max_m) for v in s.ranges_m):raise RuntimeError('Invalid LiDAR range')
        return min((v for v in s.ranges_m if math.isfinite(v)),default=s.range_max_m)


class LiveObstacleMap:
    def __init__(self,static_boxes,frame):
        self.static_boxes=list(static_boxes);self.frame=frame;self.points=[];self.sequences=set()
        self.projector=LidarRiskDetector(None,resolution_m=.1,local_frame=frame,sensor_forward_m=-.1)

    def ingest(self,scan,position,attitude,now):
        if scan is None or scan.age_s(now)>.5:raise ValueError('Stale scan')
        if scan.sequence in self.sequences:return 0
        if max(abs(attitude['roll']),abs(attitude['pitch']))>1:raise ValueError('Projection requires near-level attitude')
        found=[]
        for i,d in enumerate(scan.ranges_m):
            if math.isnan(d) or d<scan.range_min_m:raise ValueError('Invalid scan range')
            if not math.isfinite(d) or d>min(3.,scan.range_max_m):continue
            _,_,n,e=self.projector._global_cell(d,scan.angle_at(i),position['north'],position['east'],attitude['yaw'])
            point=self.frame.to_map(e,n)
            if not all(.5<=v<=19.5 for v in point):continue
            if any(inside(point,b,.15) for b in self.static_boxes):continue
            found.append(point)
        self.sequences.add(scan.sequence)
        # Do not form a new obstacle from a single isolated return.
        if len(found)<3:return 0
        self.points.extend(found);return len(found)

    def boxes(self):
        if not self.points:return []
        xs,ys=zip(*self.points)
        return [dict(name='observed_unknown_returns',x0=min(xs)-.10,x1=max(xs)+.10,y0=min(ys)-.10,y1=max(ys)+.10,z0=0.,z1=100.)]

    def route_blocked(self,points):
        boxes=self.boxes()
        for a,b in zip(points,points[1:]):
            steps=max(1,math.ceil(math.dist(a,b)/.05))
            for i in range(steps+1):
                p=tuple(x+(y-x)*i/steps for x,y in zip(a,b))
                if any(inside(p,box,1.8) for box in boxes):return True
        return False

    def replan(self,start,goal):
        if not self.points:raise ValueError('No sensor obstacle evidence')
        boxes=self.static_boxes+self.boxes()
        route=build(boxes,start=start,goal=goal)
        # Grid snapping must not create an unchecked connector from actual pose.
        for i in range(11):
            p=tuple(a+(b-a)*i/10 for a,b in zip(start,route[0]))
            if any(inside(p,b,1.8) for b in boxes):raise ValueError('Unsafe start connector')
        return [tuple(start)]+route[1:]
