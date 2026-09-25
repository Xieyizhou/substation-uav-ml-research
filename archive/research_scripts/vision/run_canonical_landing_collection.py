#!/usr/bin/env python3
"""Gate small canonical collection on an actual cancellation/landing flight."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256
from src.planner.astar_grid import astar,simplify_grid_path
from src.planner.obstacle_config import build_obstacle_map
from src.vision.collection.route import VisualRoute,ObservationWaypoint
from scripts.vision.run_hard_example_flight import _events,_wait_event_fields,_wait_for_previous_launcher,_terminate
from scripts.vision.run_background_calibration_flight import stop_owned_group

def save(path,record):
    record['identity']=object_sha256(record)
    with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')

def main():
    out=ROOT/'data/research/run19-product-scope-v1/landing-and-canonical-pilot-v1'
    out.mkdir(exist_ok=False)
    config_path=ROOT/'config/substation_obstacles.json'
    config=json.loads(config_path.read_text());nav=build_obstacle_map(config)
    env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1','GZ_IP':'127.0.0.1','GZ_PARTITION':'substation_uav','MPLCONFIGDIR':'/tmp/matplotlib-cache'}
    python=str(ROOT/'.venv/bin/python');results=[]
    plans=[('cancel-regression',None,[(1,1)],[(1.5,5.5)]),('cabinet-contrast',791001,[(2,12),(2,16)],[(5,14),(9,16)]),('switchgear-contrast',791002,[(3,1),(3,2)],[(10,2),(10,2)])]
    for name,seed,points,targets in plans:
        folder=out/name;folder.mkdir();events=folder/'flight-events.jsonl';launcher=flight=raw=None;error=None;passed=False;rc=None;cancel_time=None
        waypoints=[];current=(0,0)
        for i,(cell,target) in enumerate(zip(points,targets)):
            path=astar(current,cell,nav['inflated_blocking_cells'],20,20)
            if not path:raise ValueError(f'Unreachable {cell}')
            yaw=math.degrees(math.atan2(target[0]-cell[0]-.5,target[1]-cell[1]-.5))%360
            waypoints.append(ObservationWaypoint('complete_distant' if i==0 else f'view_{i}','cruise_distant',cell[0]+.5,cell[1]+.5,1.5,yaw,60.0,'unreviewed',tuple(simplify_grid_path(path)[1:])))
            current=cell
        back=astar(current,(0,0),nav['inflated_blocking_cells'],20,20)
        if not back:raise ValueError('Unreachable return')
        route=VisualRoute(name,'substation_simple_v4',None,None,(0,0),tuple(waypoints),tuple(simplify_grid_path(back)[1:])).to_record()
        route_path=folder/'route.json'
        with route_path.open('x') as f:json.dump(route,f,indent=2)
        print('START',name,flush=True)
        with (folder/'simulator.log').open('x') as simlog,(folder/'flight.log').open('x') as flightlog,(folder/'raw-truth.jsonl').open('x') as rawlog,(folder/'raw-truth.stderr').open('x') as rawerr:
            try:
                _wait_for_previous_launcher(ROOT/'.runtime/px4_launcher.pid')
                launcher=subprocess.Popen([python,'main.py','map','start','simple','--vehicle-model','x500_research','--display-mode','headless'],cwd=ROOT,env=env,stdout=simlog,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(140)
                if launcher.poll() is not None:raise RuntimeError('Launcher exited before prearm')
                snapshots={}
                for filename,source in [('world.sdf',ROOT/'.runtime/worlds/substation_simple_research.sdf'),('obstacles.json',config_path),('simulator_labels.py',ROOT/'src/vision/collection/simulator_labels.py')]:
                    payload=source.read_bytes()
                    with (folder/filename).open('xb') as f:f.write(payload)
                    snapshots[filename]={'source':str(source),'sha256':hashlib.sha256(payload).hexdigest()}
                save(folder/'provenance.json',{'schema_version':1,'map_id':'simple','seed':seed,'split':'development' if seed else 'safety_regression_only','files':snapshots,'route_identity':route['route_identity_sha256'],'scope':'Unmodified canonical map; cabinet remains non-target. Raw truth is offline-only.','sensor_include':'model://x500_research','training_admitted':False})
                with (folder/'prearm.log').open('x') as log:
                    probe=subprocess.run([python,'scripts/flight/probe_rgbd_native.py','--pairs','100','--timeout','45','--output',str(folder/'prearm.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=100)
                if probe.returncode:raise RuntimeError('Prearm failed')
                raw=subprocess.Popen(['gz','topic','-e','--json-output','-t','/research_camera/boxes'],cwd=ROOT,env=env,stdout=rawlog,stderr=rawerr)
                flight=subprocess.Popen([python,'scripts/flight/run_task.py','run','fly_round_trip','--','--obstacle-config',str(folder/'obstacles.json'),'--visual-route',str(route_path),'--visual-mission-events',str(events)],cwd=ROOT,env=env,stdout=flightlog,stderr=subprocess.STDOUT)
                _wait_event_fields(events,'yaw_settled',flight,240,waypoint_name='complete_distant')
                if seed is None:
                    cancel_time=time.monotonic_ns();flight.send_signal(signal.SIGINT);rc=flight.wait(timeout=120)
                    rows=_events(events);landed=[r for r in rows if r.get('event_type')=='landing_confirmed' and r.get('host_monotonic_ns',0)>=cancel_time]
                    passed=bool(landed) and any(r.get('event_type')=='mission_failed' for r in rows) and not any(r.get('event_type') in ('mission_completed','collision') for r in rows)
                    if not passed:raise RuntimeError('Cancellation recovery did not produce required landing/failure evidence')
                else:
                    with (folder/'collector.log').open('x') as log:
                        cr=subprocess.run([python,'scripts/vision/collect_hard_examples.py','--map-id','simple','--seed',str(seed),'--split','development','--frames','30','--emit-stride','30','--timeout','45','--output',str(folder/'collection')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    if cr.returncode:raise RuntimeError('Collection failed')
                    rc=flight.wait(timeout=360);rows=_events(events)
                    passed=rc==0 and any(r.get('event_type')=='landing_confirmed' for r in rows) and not any(r.get('event_type') in ('mission_failed','collision') for r in rows)
                    if not passed:raise RuntimeError('Collection flight safety evidence failed')
            except (Exception,KeyboardInterrupt) as exc:
                error=f'{type(exc).__name__}: {exc}';passed=False
            finally:
                if flight is not None and flight.poll() is None:
                    try:_terminate(flight,timeout=180)
                    except Exception as exc:error=f'{error}; recovery error: {exc}';passed=False
                if raw is not None:
                    _terminate(raw,timeout=10)
                landed=any(r.get('event_type')=='landing_confirmed' for r in _events(events))
                retained=flight is not None and not landed
                if not retained:stop_owned_group(launcher)
                else:passed=False;error=f'{error}; simulator retained: landing unconfirmed'
        rawpath=folder/'raw-truth.jsonl'
        if seed is not None and passed and rawpath.stat().st_size==0:passed=False;error='Raw truth capture empty'
        receipt={'schema_version':1,'status':'complete' if passed else 'blocked','stage':name,'seed':seed,'map_id':'simple','landing_confirmed':landed,'flight_returncode':rc,'cancel_requested_monotonic_ns':cancel_time,'simulator_retained':retained,'error':error,'raw_truth_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),'training_admitted':False}
        save(folder/'run-receipt.json',receipt);results.append(receipt);print(json.dumps(receipt),flush=True)
        if not passed:break
    save(out/'batch-receipt.json',{'schema_version':1,'status':'complete_pending_data_audit' if len(results)==3 and all(r['status']=='complete' for r in results) else 'blocked','runs':results,'training_started':False})
    return 0 if len(results)==3 and all(r['status']=='complete' for r in results) else 2
if __name__=='__main__':raise SystemExit(main())
