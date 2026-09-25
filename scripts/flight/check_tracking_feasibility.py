"""Offline acceptance of existing frozen paths under their tracking allowance."""
import json
from pathlib import Path
from scripts.flight.prepare_avoidance_route import build
from scripts.flight.fly_px4_shadow_hover import ROOT
from src.flight.tracking_envelope import check_route
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'tracking-envelope-feasibility-v1'


def main():
    run=BASE/'replan-capture-v4-baseline-1'
    pp=run/'protocol.json';rp=run/'route.json';ep=run/'evidence-replay.json'
    protocol=json.loads(pp.read_text());route=json.loads(rp.read_text());replay=json.loads(ep.read_text())
    for record in (protocol,replay):
        for path,digest in record['inputs'].items():
            if file_sha256(path)!=digest:raise ValueError('Changed input: '+path)
    if replay['status']!='raw_scan_pose_maps_replayed':raise ValueError('Unverified scan evidence')
    sources=[pp,rp,ep,Path(__file__).resolve(),ROOT/'src/flight/tracking_envelope.py',ROOT/'scripts/flight/prepare_avoidance_route.py',ROOT/'src/planner/astar_grid.py']
    findings=[]
    # Zero ADDITIONAL map uncertainty is an optimistic lower bound. Observed
    # boxes retain their existing .10m padding; no points/bounds are removed.
    boxes=protocol['static_boxes']
    findings.append(dict(unit='initial_route_static_only',**check_route(protocol['initial_route'],boxes)))
    for i,plan in enumerate(route['replans'],1):
        mp=run/f'map-before-replan-{i}.json';m=json.loads(mp.read_text());sources.append(mp)
        findings.append(dict(unit=f'replan_{i}',**check_route(plan['points'],boxes+m['boxes'])))
    end=run/'event-map-004.json';snapshot=json.loads(end.read_text());sources.append(end)
    remaining=[snapshot['route'][snapshot['index']]]+snapshot['route'][snapshot['index']+1:]
    findings.append(dict(unit='last_active_segment_after_map_growth',**check_route(remaining,boxes+snapshot['boxes'])))
    alternatives=[]
    for name,b in [('static_only',boxes),('final_observed_map',boxes+snapshot['boxes'])]:
        try:
            path=build(b,start=protocol['initial_route'][0],goal=protocol['goal'],clearance=1.98)
            result=check_route(path,b)
            alternatives.append(dict(name=name,path=path,check=result))
        except ValueError as exc:
            alternatives.append(dict(name=name,path=None,error=str(exc),interpretation='No accepted path from this conservative lattice search; not a proof that every continuous path is impossible'))
    # Reject unless both geometric tracking and map uncertainty are justified.
    # This dataset provides no certified dynamic map-error bound.
    geometry_passed=all(r['passed'] for r in findings)
    OUT.mkdir(exist_ok=False)
    write_record(OUT/'protocol.json',dict(version='tracking-envelope-feasibility-v1',clearance_m=1.8,tracking_error_m=.18,additional_map_uncertainty_lower_bound_m=0.,existing_observed_padding_retained_m=.1,map_uncertainty_bound_verified=False,can_launch_flight=False,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in sources}))
    write_record(OUT/'completion.json',dict(status='blocked',existing_geometry_passed=geometry_passed,flight_authorized=False,sandbox_integration_allowed=False,findings=findings,alternative_searches=alternatives,blockers=['Frozen routes fail tracking-envelope checks'] if not geometry_passed else ['Map uncertainty bound remains unverified'],map_uncertainty_bound_verified=False,training_admitted=False,promotable=False,inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}))
    print(json.dumps(dict(status='blocked',flight_authorized=False,findings=findings,alternative_searches=alternatives),indent=2))


if __name__=='__main__':main()
