"""Verify actual training progress and frozen preflight before recording start."""
from datetime import datetime,timezone
from pathlib import Path
import yaml
from scripts.vision.routed_fixed_bn_control import OUT,KEYS,SOURCE,SOURCE_KEYS,checked,freeze
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    p=freeze();paths=[OUT/'protocol.json',OUT/'entry-ready.json',OUT/'launch-readiness.json']
    ready,gate=map(checked,paths[1:])
    if ready['actual_draws_verified']!=8640 or gate['test_returncode']!=0 or gate['integrity']['pinned_files_verified']!=40:raise ValueError('Gate incomplete')
    records=[]
    for k,o in zip(KEYS,SOURCE_KEYS):
        q=OUT/'actual-preflight'/f'{k}.json';r=checked(q)
        if r['actual']!=p['schedules'][k] or r['tensor_records']!=checked(SOURCE/'actual-preflight'/f'{o}.json')['tensor_records']:raise ValueError('Input mismatch')
        paths.append(q)
    for k in KEYS[:2]:
        attempts=sorted((OUT/'training'/k).glob('attempt-*'))
        if not attempts:raise ValueError('No actual attempt')
        a=attempts[-1]
        if (a/'failure.json').exists():raise ValueError('Failed attempt')
        proof=a/'first-ten-steps.json';v=checked(proof)
        if v['status']!='real_fixed_bn_steps_and_feature_adaptation_verified' or len(v['steps'])!=10:raise ValueError('Missing actual optimizer evidence')
        ap=a/'run/args.yaml';cfg=yaml.safe_load(ap.read_text())
        for name,expected in dict(epochs=48,lr0=.00025,lrf=1,batch=6,nbs=6,optimizer='AdamW',imgsz=640,warmup_epochs=0).items():
            if cfg[name]!=expected:raise ValueError('Actual config drift '+name)
        if cfg['model']!=p['initialization']['path'] or cfg.get('freeze') not in (None,0):raise ValueError('Initialization or parameter freeze drift')
        records.append(dict(key=k,minimum_optimizer_steps=10,proof=str(proof.resolve())))
        paths.extend([proof,ap])
    paths.append(Path(__file__))
    dest=OUT/'training-start-receipt.json'
    if dest.exists():return checked(dest)
    return write_record(dest,dict(status='two_seeds_real_steps_verified_third_and_evaluation_queued',
        records=records,observed_at=datetime.now(timezone.utc).isoformat(),training_complete=False,
        model_passed=False,training_admitted=False,promotable=False,
        inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
