"""Verify actual regional learning rates and first epoch before reporting start."""
import csv
from datetime import datetime,timezone
from pathlib import Path
import yaml
from scripts.vision.routed_small_backbone_control import OUT,KEYS,checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/n for n in ('protocol.json','entry-ready.json','launch-readiness.json')]
    p,e,l=map(checked,paths)
    if e['actual_draws_verified']!=8640 or l['test_returncode']!=0 or l['integrity']['pinned_files_verified']!=40:raise ValueError('Incomplete preflight')
    records=[]
    for key in KEYS[:2]:
        attempts=sorted((OUT/'training'/key).glob('attempt-*'))
        if not attempts:raise ValueError('No real attempt')
        a=attempts[-1]
        if (a/'failure.json').exists():raise ValueError('Failed attempt')
        q=a/'first-ten-steps.json';proof=checked(q)
        cfg=yaml.safe_load((a/'run/args.yaml').read_text())
        if cfg['model']!=p['initialization']['path'] or cfg['epochs']!=48 or cfg['lr0']!=.00025 or cfg['lrf']!=1. or cfg['batch']!=6 or cfg['nbs']!=6 or cfg.get('freeze') not in (None,0):raise ValueError('Actual config mismatch')
        if len(proof['steps'])!=10:raise ValueError('No ten-step evidence')
        for i,s in enumerate(proof['steps']):
            if s['step']!=i+1 or s['threads']!=4 or set(s['regions'])!={'backbone','neck_head'}:raise ValueError('Runtime mismatch')
            if any(abs(lr-(.000025 if r=='backbone' else .00025))>1e-12 for lr,r in zip(s['lrs'],s['regions'])):raise ValueError('LR mismatch')
        with (a/'run/results.csv').open() as f:rows=list(csv.DictReader(f))
        if not rows or int(rows[0]['epoch'])!=1:raise ValueError('No complete epoch')
        records.append(dict(key=key,verified_steps=10,first_epoch=rows[0]));paths += [q,a/'run/args.yaml']
    paths.append(Path(__file__))
    return write_record(OUT/'training-start-receipt.json',dict(status='two_seeds_training_verified_third_and_evaluation_queued',records=records,observed_at=datetime.now(timezone.utc).isoformat(),training_complete=False,model_passed=False,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
