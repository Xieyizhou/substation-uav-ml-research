"""Verify actual updates and immutable frozen-state evidence before reporting start."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import yaml
from scripts.vision.routed_backbone_control import OUT,KEYS,FROZEN,checked
from src.ml.artifacts import file_sha256,object_sha256
from src.vision.canonical.plan import write_record

def run():
    p=checked(OUT/'protocol.json');ready=checked(OUT/'entry-ready.json');launch=checked(OUT/'launch-readiness.json')
    model=checked(OUT/'model-preflight.json');checked(OUT/'pretraining-completion.json')
    if ready['actual_draws_verified']!=8640 or launch['test_returncode']!=0 or not launch['integrity']['integrity_passed'] or launch['integrity']['pinned_files_verified']!=40:
        raise ValueError('Launch gate incomplete')
    paths=[OUT/'protocol.json',OUT/'entry-ready.json',OUT/'launch-readiness.json',OUT/'model-preflight.json',OUT/'pretraining-completion.json',OUT/'research-plan-zh.md',Path(__file__)]
    for key in KEYS:
        q=OUT/'actual-preflight'/f'{key}.json';r=checked(q)
        if r['actual']!=p['schedules'][key] or len(r['tensor_records'])!=480:raise ValueError('Preflight incomplete')
        paths.append(q)
    records=[]
    for key in KEYS[:2]:
        attempts=sorted((OUT/'training'/key).glob('attempt-*'))
        if not attempts:raise ValueError('No actual training attempt')
        attempt=attempts[-1]
        if (attempt/'failure.json').exists():raise ValueError('Latest attempt failed')
        proof=checked(attempt/'first-ten-steps.json');a=yaml.safe_load((attempt/'run/args.yaml').read_text())
        with (attempt/'run/results.csv').open() as f:epochs=list(csv.DictReader(f))
        if not epochs or int(epochs[0]['epoch'])!=1:raise ValueError('No complete real first epoch')
        if a['freeze']!=FROZEN or a['lr0']!=.00025 or a['epochs']!=48 or a['nbs']!=6 or a['batch']!=6 or a['optimizer']!='AdamW' or a['model']!=p['initialization']['path']:
            raise ValueError('Actual training config mismatch')
        if proof['status']!='ten_real_updates_backbone_fixed_neck_head_changed' or proof['frozen_sha256']!=model['frozen_initial_sha256'] or len(proof['steps'])!=10:
            raise ValueError('No verified real updates')
        if any(s!={'step':i+1,'backbone_unchanged':True,'bn_fixed':True,'actual_threads':4,'ema_frozen_unchanged':True} for i,s in enumerate(proof['steps'])):
            raise ValueError('Runtime freeze/thread guard failed')
        records.append(dict(key=key,first_epoch=epochs[0],snapshot_sha256=object_sha256(epochs[0]),actual_verified_steps=10,mutable_log_source=str(attempt/'run/results.csv')))
        paths += [attempt/'first-ten-steps.json',attempt/'run/args.yaml']
    return write_record(OUT/'training-start-receipt.json',dict(status='two_seeds_verified_training_third_seed_and_evaluation_queued',observed_at=datetime.now(timezone.utc).isoformat(),records=records,
        training_complete=False,model_passed=False,visual_review_pending=True,training_admitted=False,promotable=False,
        inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
