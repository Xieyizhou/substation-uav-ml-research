"""Reconstruct every accepted endpoint from archived scans and paired poses."""
import argparse
import json
from pathlib import Path
from src.flight.clearance_route import ClearanceMap
from src.planner.local_frame import LocalFrame
from src.sensors.types import LaserScanFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record


def replay(out):
    protocol=json.loads((out/'protocol.json').read_text())
    journal=out/'scan-pose.jsonl';mapper=None;count=0
    with journal.open() as stream:
        for line in stream:
            r=json.loads(line);frame=LocalFrame.from_mapping(r['local_frame'])
            if mapper is None:mapper=ClearanceMap(protocol['static_boxes'],frame)
            if mapper.frame!=frame:raise ValueError('Frame changed within evidence')
            scan=LaserScanFrame.from_record(r['scan']);added=0
            expected=r['yaw_rate']<5 and max(abs(r['attitude'][1][k]) for k in ('roll','pitch'))<=1 and max(abs(scan.received_monotonic_s-r[k][0]) for k in ('local','attitude'))<=.033334
            if expected!=r['map_eligible']:raise ValueError('Mapping eligibility mismatch')
            if expected:added=mapper.ingest(scan,r['local'][1],r['attitude'][1],r['now'])
            if added!=r['new_returns']:raise ValueError('Endpoint count mismatch')
            count+=1
    if mapper is None:raise ValueError('Empty scan evidence')
    snapshots=sorted(out.glob('event-map-*.json'))+sorted(out.glob('map-before-replan-*.json'))
    for p in snapshots:
        r=json.loads(p.read_text());points=r['points']
        if points!=[list(v) for v in mapper.points[:len(points)]]:raise ValueError('Map is not a replayable endpoint prefix: '+p.name)
        check=ClearanceMap(protocol['static_boxes'],mapper.frame);check.points=points
        if check.boxes()!=r['boxes']:raise ValueError('Box mismatch: '+p.name)
    inputs=[journal,out/'protocol.json',Path(__file__).resolve(),Path('src/flight/clearance_route.py').resolve(),Path('src/flight/live_obstacle_map.py').resolve()]+snapshots
    write_record(out/'evidence-replay.json',dict(status='raw_scan_pose_maps_replayed',scan_pairs=count,accepted_endpoints=len(mapper.points),snapshots=len(snapshots),training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in inputs}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();replay(a.directory.resolve())
