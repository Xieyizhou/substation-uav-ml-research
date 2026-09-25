"""Read-only flight analysis; write separate diagnostics, never rewrite trials."""
import json
import math
from pathlib import Path
from scripts.flight.replan_repeat_campaign import BASE, OUT, validate_hashes
from src.flight.live_obstacle_map import LiveObstacleMap
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record


def main():
    campaign=json.loads((OUT/'completion.json').read_text())
    validate_hashes(campaign)
    validate_hashes(json.loads((OUT/'protocol.json').read_text()))
    records=[];files=[OUT/'protocol.json',OUT/'completion.json',Path(__file__).resolve()]
    for unit in campaign['units']:
        out=BASE/unit['directory']
        protocol=json.loads((out/'protocol.json').read_text());validate_hashes(protocol)
        route=json.loads((out/'route.json').read_text());runtime=json.loads((out/'runtime/receipt.json').read_text())
        files.extend([out/'protocol.json',out/'route.json',out/'runtime/receipt.json',out/'pillar.sdf',out/'pillar-receipt.json'])
        rows=route['samples'];maps=[]
        for i,plan in enumerate(route['replans'],1):
            mp=out/f'map-before-replan-{i}.json';m=json.loads(mp.read_text());files.append(mp)
            mapper=LiveObstacleMap(protocol['static_boxes'],LocalFrame());mapper.points=m['points']
            replay=mapper.replan(plan['points'][0],protocol['goal'])
            if [list(v) for v in replay]!=plan['points']:raise ValueError('Recorded map cannot reproduce route')
            maps.append(dict(number=i,boxes=m['boxes'],point_count=len(m['points']),replayed=True))
        records.append(dict(directory=out.name,expected=unit['expected'],passed=unit['passed'],flight_error=route['error'],runtime_error=runtime['error'] if 'error' in runtime else None,landing_confirmed=runtime['landing_confirmed'],final_armed=runtime['final_armed'],owned_processes_exited=runtime['owned_processes_exited'],duration_s=rows[-1]['monotonic']-rows[0]['monotonic'],max_cross_track_m=max(r['cross_track_m'] for r in rows),min_lidar_m=min(r['lidar_m'] for r in rows),max_speed_m_s=max(math.hypot(r['position']['vn'],r['position']['ve']) for r in rows),obstacle_stop_events=sum(e['event']=='obstacle_detected_stop_requested' for e in route['events']),completed_replans=len(route['replans']),maps=maps))
    write_record(OUT/'diagnosis.json',dict(status='sandbox_integration_blocked',records=records,limitation='Last budget-exhaustion map was not saved by the frozen runner. Its exact triggering boundary cannot be reconstructed; do not infer a collision or attribute all growth to noise.',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in files}))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
