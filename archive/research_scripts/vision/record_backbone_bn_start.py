"""Record actual optimizer progress, not merely process launch."""
import csv
from datetime import datetime,timezone
from pathlib import Path
import yaml
from scripts.vision.routed_backbone_bn_control import OUT,KEYS,checked,old
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/n for n in ('protocol.json','entry-ready.json','model-preflight.json','launch-readiness.json','pretraining-completion.json')]
    p,r,m,l,_=map(checked,paths)
    if r['actual_draws_verified']!=8640 or l['test_returncode']!=0 or l['integrity']['pinned_files_verified']!=40:raise ValueError('Launch incomplete')
    records=[]
    for key in KEYS[:2]:
        attempts=sorted((OUT/'training'/key).glob('attempt-*'))
        if not attempts:raise ValueError('Not started')
        a=attempts[-1]
        if (a/'failure.json').exists():raise ValueError('Failed attempt')
        proof=checked(a/'first-ten-steps.json');cfg=yaml.safe_load((a/'run/args.yaml').read_text())
        with (a/'run/results.csv').open() as f:rows=list(csv.DictReader(f))
        if not rows or int(rows[0]['epoch'])!=1:raise ValueError('No complete first epoch')
        if cfg['freeze']!=old.FROZEN or cfg['lr0']!=.00025 or cfg['epochs']!=48 or cfg['batch']!=6 or cfg['nbs']!=6 or cfg['model']!=p['initialization']['path']:raise ValueError('Actual config mismatch')
        steps=proof['steps']
        if len(steps)!=10 or proof['status']!='ten_real_updates_fixed_parameters_live_bn_verified':raise ValueError('Incomplete real updates')
        previous=m['initial_bn_sha256']
        for i,s in enumerate(steps):
            if s['step']!=i+1 or s['fixed_sha256']!=m['fixed_sha256'] or s['bn_sha256']==previous or s['actual_threads']!=4 or not s['ema_exact']:raise ValueError('Runtime invariant failed')
            previous=s['bn_sha256']
        records.append(dict(key=key,verified_steps=10,first_epoch=rows[0]))
        paths += [a/'first-ten-steps.json',a/'run/args.yaml']
    paths.append(Path(__file__))
    return write_record(OUT/'training-start-receipt.json',dict(status='two_seeds_real_training_verified_third_and_evaluation_queued',records=records,observed_at=datetime.now(timezone.utc).isoformat(),training_complete=False,model_passed=False,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
