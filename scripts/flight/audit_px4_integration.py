"""Audit bounded local PX4 SITL integration; does not launch or arm."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.flight.fly_px4_shadow_hover import ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'integration-closeout-001'

def checked(path):
    record=json.loads(path.read_text())
    for p,digest in record.get('inputs',{}).items():
        if file_sha256(p)!=digest:raise ValueError('Evidence changed: '+p)
    return record

def main():
    requirements=[('short-route-001','short_route_verified'),('obstacle-stop-001','bounded_obstacle_stop_verified'),('lidar-registration-001','static_lidar_registration_verified'),('pillar-route-002','bounded_static_pillar_route_verified')]
    paths=[];checks=[]
    for directory,status in requirements:
        p=BASE/directory/'completion.json';record=checked(p)
        if record['status']!=status:raise ValueError('Not completed: '+directory)
        paths.append(p);checks.append(dict(requirement=directory,status=status))
    flight=checked(BASE/'pillar-route-002/completion.json')
    runtime=checked(BASE/'pillar-route-002/runtime/receipt.json')
    protocol=checked(BASE/'pillar-route-002/protocol.json')
    model=checked(BASE/'pillar-route-002/runtime/vision/model.json')
    diagnostic=checked(BASE/'pillar-route-002/moving-alignment-diagnostic.json')
    if model['control_authority']!='none' or model['key']!='material-routed-480-7':raise ValueError('Unexpected model/control role')
    if not runtime['owned_processes_exited'] or runtime['final_armed'] is not False:raise ValueError('Cleanup not confirmed')
    modules=['test_dynamic_replanning','test_replanning_controller','test_flight_runtime','test_pillar_route','test_vehicle_envelope','test_avoidance_geometry','test_local_frame','test_sensor_contracts','test_astar_25d','test_obstacle_stop','test_short_route','test_px4_hover_trial','test_moving_alignment']
    result=subprocess.run([sys.executable,'-m','unittest',*['tests.'+m for m in modules]],capture_output=True,text=True,timeout=60)
    OUT.mkdir(exist_ok=False);(OUT/'tests.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Relevant regressions failed; see tests.log')
    paths.extend([BASE/'pillar-route-002/runtime/receipt.json',BASE/'pillar-route-002/protocol.json',BASE/'pillar-route-002/runtime/vision/model.json',BASE/'pillar-route-002/moving-alignment-diagnostic.json',OUT/'tests.log',Path(__file__).resolve()])
    paths.extend(ROOT/'tests'/f'{m}.py' for m in modules)
    write_record(OUT/'completion.json',dict(status='bounded_local_px4_sitl_integration_verified',scope='existing model as read-only sidecar; actual offboard route, obstacle braking, known static pillar avoidance, landing and cleanup',checks=checks,model=model['key'],route_metrics=flight['metrics'],moving_alignment_unknowns=diagnostic['unknown'],limitations=['Not physical flight authorization','Not dynamic obstacle replanning certification','Vision has no flight-control authority','No claim of model training admission or promotion','Moving interpolation remains diagnostic; six unmatched samples retained','Original 28.2 m wall route remains unapproved'],training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print('bounded_local_px4_sitl_integration_verified')

if __name__=='__main__':main()
