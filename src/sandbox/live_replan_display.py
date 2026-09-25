"""Bounded telemetry tail for display, never a control or acceptance input."""
import json
import math
from pathlib import Path
import time
from src.sandbox.live_replan_gate import BASE
from src.sandbox.live_replan_stop import validate_run_id

def flight_display(root,job):
    if not job:return None
    run_id=validate_run_id(job['scenario_id'])
    out=Path(root)/BASE/run_id;path=out/'runtime/telemetry.jsonl'
    result=dict(job_id=job['job_id'],run_id=run_id,job_state=job['state'],phase='awaiting_telemetry',local_position=None,telemetry_age_s=None,display_only=True)
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-16384));lines=stream.read(16384).decode('utf-8',errors='replace').splitlines()
        for line in reversed(lines):
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('stream')!='local':continue
            value=row['value'];age=time.monotonic()-row['monotonic']
            if not all(math.isfinite(value[k]) for k in ('north','east','down','vn','ve')) or not math.isfinite(age):break
            result.update(phase=row['phase'],local_position={k:value[k] for k in ('north','east','down')},speed_m_s=math.hypot(value['vn'],value['ve']),telemetry_age_s=max(0,age))
            break
    except (OSError,KeyError,TypeError):pass
    return result
