"""Record a verified live training start, never equate launch with completion."""
import argparse
import csv
import io
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.interleaved_small_scale_control import OUT,KEYS,prior
from scripts.vision.train_interleaved_small_scale import ready


def record(runner,workers):
    rp=ready()
    if set(workers)!=set(KEYS[:2]): raise ValueError('Expected initial seed pair')
    deps=[rp,OUT/'design.json',Path(__file__).resolve()]
    commands={}
    for pid,key in [(runner,None)]+[(pid,k) for k,pid in workers.items()]:
        command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip()
        if '-m scripts.vision.train_interleaved_small_scale --train' not in command:
            raise ValueError('Wrong or dead process')
        if key and '--worker '+key not in command: raise ValueError('Wrong worker')
        if not key and '--then-evaluate' not in command: raise ValueError('Automatic evaluation missing')
        commands[str(pid)]=command
    observations=[]
    for key,pid in workers.items():
        path=OUT/'training'/key/'attempt-001/run/results.csv'
        payload=path.read_text(); rows=list(csv.DictReader(io.StringIO(payload)))
        if not rows or int(float(rows[-1]['epoch']))<1: raise ValueError('No completed optimization epoch')
        observations.append(dict(cell=key,pid=pid,completed_epochs=int(float(rows[-1]['epoch'])),
            minimum_optimizer_steps=int(float(rows[-1]['epoch']))*10,results_snapshot=payload))
    dest=OUT/'training-launch.json'
    if dest.exists():
        r=prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest,dict(status='review_analysis_preflight_complete_training_started',
        observed_at=datetime.now(timezone.utc).isoformat(),runner_pid=runner,commands=commands,observations=observations,
        scheduled_cells=list(KEYS),automatic_evaluation_after_all_training_success=True,
        training_complete=False,evaluation_complete=False,
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--runner',type=int,required=True)
    ap.add_argument('--reference-worker',type=int,required=True); ap.add_argument('--material-worker',type=int,required=True); a=ap.parse_args()
    r=record(a.runner,{KEYS[0]:a.reference_worker,KEYS[1]:a.material_worker}); print(r['status'])
