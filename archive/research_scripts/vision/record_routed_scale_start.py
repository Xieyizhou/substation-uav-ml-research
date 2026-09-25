"""Record actual first epochs after input/label preflight, not process intent."""
import csv
from datetime import datetime,timezone
from pathlib import Path
import yaml
from scripts.vision.routed_scale_control import OUT,KEYS,checked,preview_gate
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/n for n in ('protocol.json','entry-ready.json','launch-readiness.json')]+preview_gate()
    p,e,l=map(checked,paths[:3])
    if e['actual_draws_verified']!=8640 or l['test_returncode']!=0 or l['integrity']['pinned_files_verified']!=40:raise ValueError('Gate incomplete')
    for k in KEYS:
        q=OUT/'actual-preflight'/f'{k}.json';r=checked(q)
        if r['actual']!=p['schedules'][k] or len(r['tensor_records'])!=480:raise ValueError('Preflight mismatch')
        if [v for b in r['tensor_records'] for v in b['scale_factors']]!=p['scale_factors'][k]:raise ValueError('Scale sequence mismatch')
        paths.append(q)
    records=[]
    for k in KEYS[:2]:
        attempts=sorted((OUT/'training'/k).glob('attempt-*'))
        if not attempts:raise ValueError('No actual attempt')
        a=attempts[-1]
        if (a/'failure.json').exists():raise ValueError('Attempt failed')
        ap=a/'run/args.yaml';cfg=yaml.safe_load(ap.read_text())
        if cfg['epochs']!=48 or cfg['lr0']!=.00025 or cfg['lrf']!=1 or cfg['batch']!=6 or cfg['nbs']!=6 or cfg['model']!=p['initialization']['path'] or cfg.get('freeze') not in (None,0):raise ValueError('Actual training config changed')
        with (a/'run/results.csv').open() as f:rows=list(csv.DictReader(f))
        if not rows or int(rows[0]['epoch'])!=1:raise ValueError('First epoch not complete')
        records.append(dict(key=k,completed_epochs_at_snapshot=len(rows),first_epoch=rows[0],minimum_optimizer_steps=10,mutable_log_source=str(a/'run/results.csv')));paths.append(ap)
    paths.append(Path(__file__))
    return write_record(OUT/'training-start-receipt.json',dict(status='two_seeds_first_epoch_verified_third_and_evaluation_queued',records=records,observed_at=datetime.now(timezone.utc).isoformat(),training_complete=False,model_passed=False,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
