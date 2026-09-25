"""Require actual optimizer evidence and immutable real-loader preflight."""
from datetime import datetime,timezone
from pathlib import Path
import yaml
from scripts.vision.routed_gamma_control import OUT,KEYS,SOURCE,SOURCE_KEYS,checked,freeze
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    p=freeze();paths=[OUT/'protocol.json',OUT/'entry-ready.json',OUT/'launch-readiness.json']
    ready,gate=map(checked,paths[1:])
    if ready['actual_draws_verified']!=8640 or gate['test_returncode']!=0 or gate['integrity']['pinned_files_verified']!=40:raise ValueError('Gate incomplete')
    for k,o in zip(KEYS,SOURCE_KEYS):
        q=OUT/'actual-preflight'/f'{k}.json';r=checked(q);old=checked(SOURCE/'actual-preflight'/f'{o}.json')
        if r['actual']!=p['schedules'][k] or len(r['tensor_records'])!=480:raise ValueError('Exposure mismatch')
        for j,(a,b) in enumerate(zip(r['tensor_records'],old['tensor_records'])):
            if a['images']!=b['images'] or a['routed_tensors']!=b['tensors'] or a['gamma_factors']!=p['gamma_factors'][k][j*6:j*6+6]:raise ValueError('Tensor identity mismatch')
            for f in ('cls','bboxes','batch_idx'):
                if a['tensors'][f]!=b['tensors'][f]:raise ValueError('Labels changed')
        paths.append(q)
    records=[]
    for k in KEYS[:2]:
        attempts=sorted((OUT/'training'/k).glob('attempt-*'))
        if not attempts:raise ValueError('No attempt')
        a=attempts[-1]
        if (a/'failure.json').exists():raise ValueError('Attempt failed')
        proof=a/'first-ten-steps.json';v=checked(proof)
        if v['status']!='real_gamma_training_started' or v['optimizer_steps']!=10 or v['learning_rates']!=[.00025]*10:raise ValueError('No actual optimizer evidence')
        ap=a/'run/args.yaml';cfg=yaml.safe_load(ap.read_text())
        for name,value in dict(epochs=48,lr0=.00025,lrf=1,batch=6,nbs=6,optimizer='AdamW',imgsz=640,warmup_epochs=0).items():
            if cfg[name]!=value:raise ValueError('Config drift '+name)
        if cfg['model']!=p['initialization']['path'] or cfg.get('freeze') not in (None,0):raise ValueError('Initialization drift')
        paths.extend([proof,ap]);records.append(dict(key=k,minimum_optimizer_steps=10,proof=str(proof.resolve())))
    paths.append(Path(__file__));dest=OUT/'training-start-receipt.json'
    if dest.exists():return checked(dest)
    return write_record(dest,dict(status='two_seeds_real_steps_verified_third_and_evaluation_queued',records=records,observed_at=datetime.now(timezone.utc).isoformat(),training_complete=False,model_passed=False,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
