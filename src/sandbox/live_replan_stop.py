"""Bounded cooperative landing request, restricted to this SITL workflow."""
import json
from pathlib import Path
import re
import time
from src.sandbox.live_replan_gate import BASE

ACTION='live-replan-flight'

def validate_run_id(value):
    if not isinstance(value,str) or re.fullmatch(r'sandbox-replan-v1-[0-9a-f]{32}',value) is None:
        raise ValueError('Invalid sandbox live-replan run ID')
    return value

def check_stop(out):
    if (Path(out)/'stop-requested.json').exists():raise RuntimeError('Sandbox requested controlled stop and landing')

def cooperative_stop(config,job,has_stopped,*,timeout_s=90.):
    if job.action!=ACTION:return False
    run_id=validate_run_id(job.scenario_id)
    base=(Path(config.project_root)/BASE).resolve()
    out=base/run_id
    if out.resolve().parent!=base:raise ValueError('Stop target escapes live-replan root')
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'stop-requested.json'
    if not marker.exists():
        marker.write_text(json.dumps(dict(job_id=job.job_id,action=ACTION,reason=job.error or 'operator stop',simulation_only=True)))
    deadline=time.monotonic()+timeout_s
    while not has_stopped():
        if time.monotonic()>=deadline:return False
        time.sleep(.1)
    return True
