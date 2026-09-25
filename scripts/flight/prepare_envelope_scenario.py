"""Freeze a separate feasible-world fixture; preserve original narrow-world rejection."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.flight.fly_px4_shadow_hover import ROOT
from scripts.flight.prepare_avoidance_route import collision_boxes,build
from src.flight.tracking_envelope import check_route
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'envelope-aware-scene-v1'
GOAL=(1.7,6.9)
INITIAL=[(1.5,1.5),(1.6,.9),(1.6,6.9),GOAL]
CASES={
    'baseline':dict(east=1.2,north=4.,trigger_north=.8),
    'west':dict(east=1.,north=4.,trigger_north=.8),
    'later':dict(east=1.2,north=4.,trigger_north=1.15),
}


def main():
    source=ROOT/'simulation/worlds/substation_simple.sdf'
    tree=ET.parse(source);root=tree.getroot();container=root.find("world/model[@name='substation_map']")
    changes={'transformer_1':(2.5,.5,0.),'pole_west_1':(3.,0.,0.),'fence_south':(0.,-1.,0.)}
    originals={};deltas=[]
    for name,delta in changes.items():
        nodes=container.findall(f"model[@name='{name}']")
        if len(nodes)!=1:raise ValueError('Ambiguous world object: '+name)
        node=nodes[0];pose=node.find('pose');original=pose.text;values=list(map(float,original.split()))
        originals[name]=original
        pose.text=' '.join(map(str,[values[i]+delta[i] for i in range(3)]+values[3:]))
        deltas.append(dict(object=name,original=original,revised=pose.text))
    # Verify the only XML changes are the three explicitly named poses.
    revised=ET.tostring(root)
    for name,text in originals.items():container.find(f"model[@name='{name}']/pose").text=text
    if ET.tostring(root)!=ET.tostring(ET.parse(source).getroot()):raise ValueError('Unexpected structural mutation')
    OUT.mkdir(exist_ok=False);world=OUT/'world.sdf';world.write_bytes(revised)
    boxes=collision_boxes(world);initial=check_route(INITIAL,boxes,map_uncertainty=.20)
    if not initial['passed']:raise ValueError('Initial robust route failed: '+str(initial))
    cases=[]
    for name,c in CASES.items():
        # Harness-only conservative full footprint: physical half-width .15,
        # observed padding .10 plus .20 empirical projection allowance.
        half=.45
        b=dict(name='harness_only_full_uncertain_pillar',x0=c['east']-half,x1=c['east']+half,y0=c['north']-half,y1=c['north']+half,z0=0.,z1=4.)
        path=build(boxes+[b],start=(1.6,1.1),goal=GOAL,clearance=2.18)
        result=check_route(path,boxes+[b],map_uncertainty=.20)
        if not result['passed']:raise ValueError('Robust path failed: '+name)
        cases.append(dict(name=name,fixture=c,harness_box=b,offline_path=path,check=result,maximum_nominal_displacement_m=max(math.dist(p,INITIAL[0]) for p in path),fixture_not_supplied_to_online_planner=True))
    empirical=BASE/'mapping-error-budget-v1/measurement.json';measurement=json.loads(empirical.read_text())
    if not measurement['all_measured_pairs_within_budget'] or measurement['unknown_pair_count']:
        raise ValueError('Empirical map-error budget evidence incomplete')
    narrow=BASE/'tracking-envelope-feasibility-v1/completion.json'
    if json.loads(narrow.read_text())['flight_authorized']:raise ValueError('Original narrow-world rejection lost')
    inputs=[source,world,empirical,narrow,Path(__file__).resolve(),ROOT/'src/flight/tracking_envelope.py',ROOT/'scripts/flight/prepare_avoidance_route.py']
    write_record(OUT/'protocol.json',dict(status='offline_geometry_passed_runtime_not_verified',world_changes=deltas,static_boxes=boxes,initial_route=INITIAL,goal=GOAL,cases=cases,clearance_m=1.8,tracking_error_m=.18,additional_map_budget_m=.20,planning_centerline_padding_m=2.18,original_narrow_scene_remains_rejected=True,flight_completed=False,sandbox_integration_allowed=False,scope='separate feasible simulation layout plus retained original infeasibility control; not a relabelled original-world success',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in inputs}))
    print(json.dumps(dict(initial_passed=True,cases=[dict(name=c['name'],path=c['offline_path'],displacement=c['maximum_nominal_displacement_m']) for c in cases],world_changes=deltas),indent=2))


if __name__=='__main__':main()
