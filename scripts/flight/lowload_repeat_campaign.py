"""Frozen repeat-test campaign. No Sandbox control is enabled by this tool."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.flight.check_px4_shadow import preflight

from scripts.flight.fly_px4_shadow_hover import ROOT
from scripts.flight.prepare_avoidance_route import collision_boxes, build
from scripts.flight.envelope_fixture import CASES
from src.flight.envelope_route import EnvelopeMap as LiveObstacleMap
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.flight.replay_envelope_evidence import replay

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'lowload-repeat-v1'


def validate_hashes(record):
    for path,digest in record['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Stale input: '+path)


def freeze():
    scene=BASE/'envelope-aware-scene-v1/protocol.json';frozen_scene=json.loads(scene.read_text());validate_hashes(frozen_scene)
    boxes=frozen_scene['static_boxes']
    feasibility={}
    for name,c in CASES.items():
        fixture=dict(name='harness_only_uncertain_envelope',x0=c['east']-.45,x1=c['east']+.45,y0=c['north']-.45,y1=c['north']+.45,z0=0.,z1=4.)
        try:
            path=build(boxes+[fixture],start=(1.6,1.1),goal=(1.7,6.9),clearance=2.18)
        except ValueError:
            path=None
        if (path is not None)!=(c['expected']=='goal'):
            raise ValueError('Harness scenario feasibility mismatch: '+name)
        feasibility[name]=dict(fixture=fixture,path=path,planner_receives_fixture=False)
    units=[dict(case='no_path',repeat=1,directory='lowload-repeat-v1-no_path-1',expected='safe_rejection')]
    units += [dict(case=c,repeat=n,directory=f'lowload-repeat-v1-{c}-{n}',expected='goal') for c in ('baseline','west','later') for n in (1,2)]
    sources=[Path(__file__).resolve()]+[ROOT/'scripts/flight'/name for name in ('fly_lowload_replan.py','envelope_fixture.py','audit_lowload_replan.py','lowload_trial_lifecycle.py','fly_pillar_route.py','check_final_heading.py','retest_sim_heading.py','prepare_avoidance_route.py','check_px4_shadow.py','measure_mapping_error.py')]
    sources += [ROOT/'src/flight/live_obstacle_map.py',ROOT/'src/flight/turn_hold.py',ROOT/'src/flight/envelope_route.py',ROOT/'src/flight/tracking_envelope.py',ROOT/'src/flight/replan_evidence.py',ROOT/'scripts/flight/replay_envelope_evidence.py',scene,BASE/'live-replan-turn-hold-002/completion.json']
    sources.append(ROOT/'src/flight/arrival_capture.py')
    profile=BASE/'lowload-vehicle-v1/protocol.json';profile_record=json.loads(profile.read_text());validate_hashes(profile_record)
    sources += [profile,ROOT/'src/sensors/gazebo_lidar.py',ROOT/'src/sensors/gazebo_lidar_observed.py',BASE/'sensor-load-probe-v2/rgb640/completion.json',BASE/'sensor-load-probe-v2/rgb1920/completion.json']+[Path(p) for p in profile_record['inputs']]
    OUT.mkdir(exist_ok=False)
    write_record(OUT/'protocol.json',dict(version='lowload-repeat-v1',resource_preflight_wait_s=60,change='independent RGB640 flight profile plus subscriber arrival diagnostics; unchanged lidar, safety guards and 1.8+.18+.20m planning envelope; separate feasible layout, old rejection preserved',cases=CASES,units=units,harness_feasibility=feasibility,positive_required=6,negative_required=1,max_attempts_per_unit=1,stop_on_any_unexpected_result=True,pass_rule='all six complete sensor stop/replan/resume/goal and audited landing; negative trial reproduces no path or blocked goal and safely lands; all hashes valid',scope='PX4 SITL only; stationary newly inserted pillar; no physical flight',sandbox_enable_requires_all=True,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in sources}))


def audit_rejection(out):
    protocol=json.loads((out/'protocol.json').read_text());validate_hashes(protocol)
    route=json.loads((out/'route.json').read_text());runtime=json.loads((out/'runtime/receipt.json').read_text())
    if route['status']!='failed' or not route['error'].startswith(('ValueError: No A* path found','ValueError: Goal cell')):
        raise ValueError('Not the expected no-path rejection')
    names=[e['event'] for e in route['events']]
    required=['harness_spawn_requested','obstacle_detected_stop_requested','actual_stop_confirmed']
    if any(names.count(e)!=1 for e in required) or [names.index(e) for e in required]!=sorted(names.index(e) for e in required):
        raise ValueError('Missing/ambiguous negative-test event chain')
    if route['replans'] or 'goal_reached_after_replan' in names:raise ValueError('Negative test resumed unexpectedly')
    if not runtime['landing_confirmed'] or runtime['final_armed'] is not False or not runtime['owned_processes_exited']:
        raise ValueError('Unsafe negative-test termination')
    mp=out/'map-before-replan-1.json';m=json.loads(mp.read_text())
    if m['source']!='live_lidar_only' or not m['points']:raise ValueError('Missing sensor evidence')
    mapper=LiveObstacleMap(protocol['static_boxes'],LocalFrame());mapper.points=m['points']
    try:
        mapper.replan(route['samples'][-1]['map_position'],protocol['goal'])
    except ValueError as exc:
        if not str(exc).startswith(('No A* path found','Goal cell')):raise
    else:raise ValueError('No-path rejection not reproducible')
    stops=[e for e in route['events'] if e['event']=='actual_stop_confirmed']
    if stops[0]['speed_m_s']>=protocol['stop_speed_m_s']:raise ValueError('Not stopped')
    files=[out/'protocol.json',out/'route.json',out/'runtime/receipt.json',mp,out/'pillar.sdf',out/'pillar-receipt.json',Path(__file__).resolve(),out/'scan-pose.jsonl',out/'sensor-failure.json',out/'sensor-diagnostics.json']+sorted(out.glob('event-map-*.json'))
    write_record(out/'completion.json',dict(status='expected_safe_no_path_rejection_verified',landing_confirmed=True,final_armed=False,owned_processes_exited=True,simulation_only=True,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in files}))


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());validate_hashes(protocol)
    results=[]
    for unit in protocol['units']:
        validate_hashes(protocol)
        out=BASE/unit['directory'];completion=out/'completion.json'
        expected='sensor_stop_replan_resume_goal_verified' if unit['expected']=='goal' else 'expected_safe_no_path_rejection_verified'
        print('REPEAT_UNIT',unit['directory'],flush=True)
        error=None
        try:
            if not out.exists():
                deadline=time.monotonic()+protocol['resource_preflight_wait_s']
                while True:
                    try:preflight();break
                    except OSError:
                        if time.monotonic()>=deadline:raise
                        time.sleep(1)
                command=[sys.executable,'-m','scripts.flight.fly_lowload_replan','--case',unit['case'],'--repeat',str(unit['repeat'])]
                subprocess.run(command,check=True,cwd=ROOT)
                subprocess.run(command+['--fly'],check=True,cwd=ROOT)
            if not completion.exists():
                replay(out)
                if unit['expected']=='goal':
                    subprocess.run([sys.executable,'-m','scripts.flight.audit_lowload_replan',str(out)],check=True,cwd=ROOT)
                else:audit_rejection(out)
            record=json.loads(completion.read_text());validate_hashes(record)
            validate_hashes(json.loads((out/'protocol.json').read_text()))
            if record['status']!=expected:raise ValueError('Unexpected completion status')
        except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
            error=f'{type(exc).__name__}: {exc}'
        results.append(dict(**unit,passed=error is None,error=error))
        if error is not None:break
    files=[OUT/'protocol.json']+[BASE/u['directory']/name for u in results for name in ('completion.json','route.json','runtime/receipt.json','evidence-replay.json') if (BASE/u['directory']/name).exists()]
    passed=len(results)==len(protocol['units']) and all(u['passed'] for u in results)
    write_record(OUT/'completion.json',dict(status='repeat_campaign_verified' if passed else 'blocked',sandbox_integration_allowed=passed,units=results,not_run=[u['directory'] for u in protocol['units'][len(results):]],training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in files}))
    print('CAMPAIGN_RESULT', 'passed' if passed else 'blocked',flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.freeze and args.run:parser.error('Freeze and execute separately')
    if args.freeze:freeze()
    elif args.run:sys.exit(run())
    else:
        validate_hashes(json.loads((OUT/'protocol.json').read_text()));print('Frozen campaign preflight only; no flight started')
