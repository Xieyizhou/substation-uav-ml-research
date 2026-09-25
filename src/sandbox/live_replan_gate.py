"""Read-only gate for the bounded, simulation-only live-replan workflow."""
import json
from pathlib import Path
from src.ml.artifacts import file_sha256

BASE=Path('data/research/material-shadow-v1/autonomy-avoidance-v1')
CAMPAIGN='startup-repeat-v2'
POSITIVE='sensor_stop_replan_resume_goal_verified'
NEGATIVE='expected_safe_no_path_rejection_verified'

def summary(root):
    """Cheap display only. Launch always performs require_repeat_gate again."""
    base=Path(root)/BASE;units=[]
    for case,n in [('no_path',1)]+[(c,n) for c in ('baseline','west','later') for n in (1,2)]:
        prefix='lowload-repeat-v1' if case in ('no_path','baseline') or (case,n)==('west',1) else CAMPAIGN
        path=base/f'{prefix}-{case}-{n}'/'completion.json'
        try:record=json.loads(path.read_text());status=record['status']
        except (OSError,ValueError,KeyError):status='pending_or_invalid'
        units.append(dict(case=case,repeat=n,recorded_status=status))
    try:record=json.loads((base/CAMPAIGN/'completion.json').read_text())
    except (OSError,ValueError):record={}
    ready=record.get('status')=='repeat_campaign_verified' and record.get('sandbox_integration_allowed') is True
    return dict(status='ready_for_launch_revalidation' if ready else 'locked_pending_repeat_gate',ready_for_launch_revalidation=ready,units=units,scope='PX4 SITL only; fixed baseline layout, one newly inserted stationary pillar, LiDAR control, vision read-only; RGB640 flight-only profile',note='Display reads recorded status only; all evidence hashes are checked before every launch.',training_admitted=False,promotable=False)

def validate_record(path):
    record=json.loads(Path(path).read_text())
    if not isinstance(record.get('inputs'),dict) or not record['inputs']:
        raise ValueError('Missing evidence dependencies: '+str(path))
    for source,digest in record['inputs'].items():
        if file_sha256(source)!=digest:raise ValueError('Stale evidence: '+source)
    return record

def require_repeat_gate(root):
    base=Path(root)/BASE
    completion=validate_record(base/CAMPAIGN/'completion.json')
    protocol=validate_record(base/CAMPAIGN/'protocol.json')
    expected={('no_path',1)}|{(case,n) for case in ('baseline','west','later') for n in (1,2)}
    units=protocol['units'];results=completion['units']
    if len(units)!=7 or {(u['case'],u['repeat']) for u in units}!=expected:
        raise ValueError('Repeat matrix missing or duplicated')
    if completion['status']!='repeat_campaign_verified' or completion.get('sandbox_integration_allowed') is not True:
        raise ValueError('Repeat campaign has not passed')
    if completion.get('not_run') or len(results)!=7 or any(r.get('passed') is not True for r in results):
        raise ValueError('Incomplete repeat results')
    if {(r['case'],r['repeat'],r['directory']) for r in results}!={(u['case'],u['repeat'],u['directory']) for u in units}:
        raise ValueError('Repeat result identity mismatch')
    validate_record(base/'lowload-vehicle-v1/protocol.json')
    validate_record(base/'envelope-aware-scene-v1/protocol.json')
    for unit in units:
        directory=unit['directory']
        prefix='lowload-repeat-v1' if unit['case'] in ('no_path','baseline') or (unit['case'],unit['repeat'])==('west',1) else CAMPAIGN
        if directory!=f"{prefix}-{unit['case']}-{unit['repeat']}":raise ValueError('Unexpected flight directory')
        out=base/directory
        done=validate_record(out/'completion.json')
        validate_record(out/'protocol.json')
        runtime=validate_record(out/'runtime/receipt.json')
        replay=validate_record(out/'evidence-replay.json')
        required=NEGATIVE if unit['case']=='no_path' else POSITIVE
        if done['status']!=required:raise ValueError('Unexpected flight result')
        if runtime.get('landing_confirmed') is not True or runtime.get('final_armed') is not False or runtime.get('owned_processes_exited') is not True:
            raise ValueError('Incomplete landing/cleanup evidence')
        if replay['status']!='raw_scan_pose_maps_replayed':raise ValueError('Missing raw-sensor replay')
        if required==POSITIVE:
            validate_record(out/'map-registration.json')
            vision=validate_record(out/'runtime/vision/completion.json')
            if vision['status']!='live_complete_not_flight_certified':raise ValueError('Incomplete vision evidence')
    return dict(status='repeat_gate_verified',positive_flights=6,negative_flights=1,simulation_only=True,physical_flight_certified=False,training_admitted=False,promotable=False,campaign_hash=file_sha256(base/CAMPAIGN/'completion.json'))
