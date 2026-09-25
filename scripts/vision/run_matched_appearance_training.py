"""Execute the six frozen K/L cells with process isolation and unchanged evaluation."""
import argparse
import csv
import fcntl
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import yaml
from scripts.vision.freeze_full_image_training import OUT, ROOT, NAMES, SEEDS, read, save, file_sha256, verify_tree, baseline_verify
from scripts.vision.verify_full_image_training_freeze import MODULES
from scripts.vision.exposure_protocol import verify
import scripts.vision.run_exposure_diagnosis as trainer
import scripts.vision.run_visibility_quality_training as evaluator
from scripts.vision.analyze_visibility_quality_results import paired_change

KEYS = tuple(f'{a}-300-{s}' for s in SEEDS for a in ('K','L'))
ZERO_AUG = ('mosaic','mixup','copy_paste','degrees','translate','scale','shear','perspective',
            'flipud','fliplr','hsv_h','hsv_s','hsv_v')

def validate_args(args):
    expected=dict(imgsz=640,batch=6,nbs=6,device='cpu',workers=0,optimizer='AdamW',
                  lr0=.001,lrf=1.,warmup_epochs=0,epochs=30,amp=False,deterministic=True,val=False)
    for k,v in expected.items():
        if args.get(k)!=v:raise ValueError(f'Runtime control mismatch: {k}')
    if any(args.get(k)!=0 for k in ZERO_AUG):raise ValueError('Runtime augmentation enabled')

def validate_evaluation(r, key):
    if r['status']!='complete' or r['cell']!=key or r['matching_conflicts']:
        raise ValueError('Evaluation status or matching conflict')
    if len(r['rows'])!=48 or len(r['negative_rows'])!=48:
        raise ValueError('Incomplete development evaluation')
    if len({(x['pair_id'],x['variant']) for x in r['rows']})!=48:
        raise ValueError('Duplicate paired evaluation')
    if len({(x['view_id'],x['variant']) for x in r['negative_rows']})!=48:
        raise ValueError('Duplicate no-target evaluation')
    if any(x['matching_conflict'] for x in r['rows']):raise ValueError('Per-frame matching conflict')

def retention(candidate,reference,tolerance):
    result=[]
    for variant in ('original','lighting'):
        for category in (None,*NAMES):
            a=candidate[variant] if category is None else candidate[variant]['per_class'][category]
            b=reference[variant] if category is None else reference[variant]['per_class'][category]
            delta=a['instance_recall']['mean']-b['instance_recall']['mean']
            result.append(dict(metric=f'{variant}.{category or "all"}.instance_recall',reference='matched_K-300',
                actual_delta=delta,minimum_delta=-tolerance,passed=delta>=-tolerance-1e-12))
    return result

def checked_runtime(key,p):
    cell=trainer.checked_cell(OUT/key/'completion.json',p)
    if cell['cell']!=key or cell['optimizer_steps']!=300:raise ValueError('Wrong training endpoint')
    run=Path(cell['exposure_path']).parent
    args=yaml.safe_load((run/'args.yaml').read_text());validate_args(args)
    if args['seed']!=int(key.split('-')[-1]) or args['model']!=p['controls']['initial_weights']:
        raise ValueError('Initialization or seed mismatch')
    with (run/'results.csv').open() as stream:
        rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(stream)]
    if len(rows)!=30:raise ValueError('Incomplete loss curve')
    for r in rows:
        for k,v in r.items():
            if k.startswith('train/') and not math.isfinite(float(v)):raise ValueError('Nonfinite training loss')
            if k.startswith('lr/') and abs(float(v)-.001)>1e-12:raise ValueError('Nonconstant learning rate')
    if not any(k.startswith('lr/') for k in rows[0]):raise ValueError('Missing LR curve')
    return cell

def prepare_execution():
    ep=OUT/'execution.json'
    if ep.exists():verify_tree(ep);return read(ep)
    verify_tree(OUT/'freeze-validation.json');p=read(OUT/'protocol.json')
    if p['status']!='frozen_design_execution_preflight_required' or set(p['schedules'])!=set(KEYS):
        raise ValueError('Unexpected frozen experiment')
    if p['controls']['lr0']!=.001 or p['steps']!=300:raise ValueError('Unexpected controls')
    mods=(*MODULES,'tests.test_matched_appearance_training')
    result=subprocess.run([sys.executable,'-m','unittest',*mods],cwd=ROOT,capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    import torch,ultralytics
    deps=[OUT/'protocol.json',OUT/'freeze-validation.json',Path(__file__),
          ROOT/'scripts/vision/run_exposure_diagnosis.py',ROOT/'scripts/vision/run_visibility_quality_training.py',
          ROOT/'scripts/vision/run_fixed_budget_diagnosis.py',ROOT/'scripts/vision/finalize_exposure_diagnosis.py',
          ROOT/'scripts/vision/analyze_visibility_quality_results.py',ROOT/'scripts/vision/exposure_protocol.py',
          *[ROOT/(m.replace('.','/')+'.py') for m in mods]]
    return save(ep,dict(status='execution_preflight_passed',cells=list(KEYS),protocol_identity=p['identity'],
        python_version=sys.version,torch_version=torch.__version__,ultralytics_version=ultralytics.__version__,
        regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        retry_policy='At most 3 isolated worker attempts per cell. Failure stops this invocation; valid completed training may be reused only after checks. No unverifiable optimizer resume.',
        evaluation='Fixed development only; formal .37 and diagnostic .001; no new visual review decisions generated.',
        inputs={str(x):file_sha256(x) for x in deps}))

def worker(key):
    verify_tree(OUT/'execution.json');p=read(OUT/'protocol.json')
    if key not in KEYS:raise ValueError('Unknown cell')
    trainer.OUT=OUT;evaluator.OUT=OUT
    # Retain the original protocol identity; only adapt its dataset lookup at the call boundary.
    if not (OUT/key/'completion.json').exists() and len(list((OUT/key).glob('attempt-*')))>=3:
        raise ValueError('Training attempt budget exhausted')
    trainer.train(key,{**p,'datasets':{key.split('-')[0]:p['datasets'][key]}})
    checked_runtime(key,p)
    verify_tree(OUT/'execution.json')
    r=evaluator.evaluate(key,p);validate_evaluation(r,key)
    rp=OUT/key/'runtime-validation.json'
    if rp.exists():verify_tree(rp)
    else:
        paths=[OUT/'execution.json',OUT/key/'completion.json',OUT/f'evaluation-{key}.json']
        save(rp,dict(status='complete',cell=key,optimizer_steps=300,actual_exposure_count=1800,
            runtime_controls_verified=True,loss_curve_verified=True,
            inputs={str(x):file_sha256(x) for x in paths}))
    print('CELL_EVALUATED',key,r['negative_summary'],flush=True)

def stop_process(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def run_worker(key):
    directory=OUT/'execution-attempts'/key;directory.mkdir(parents=True,exist_ok=True)
    attempts=sorted(directory.glob('attempt-*'))
    if len(attempts)>=3:raise ValueError('Worker attempt budget exhausted')
    run=directory/f'attempt-{len(attempts)+1:03}';run.mkdir()
    proc=None
    try:
        with (run/'worker.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.run_matched_appearance_training','--worker',key],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            while True:
                try:code=proc.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    print('RUNNING',key,'seconds',round(time.monotonic()-started),flush=True)
                    if time.monotonic()-started>21600:raise TimeoutError('Six-hour cell walltime limit')
            if code:raise RuntimeError(f'Worker exited {code}; see {run}/worker.log')
        save(run/'result.json',dict(status='complete',cell=key,returncode=code,
             inputs={str(run/'worker.log'):file_sha256(run/'worker.log')}))
    except BaseException:
        if proc is not None:stop_process(proc)
        save(run/'failure.json',dict(status='failed',cell=key,error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None))
        raise

def finalize(p):
    dest=OUT/'completion.json'
    if dest.exists():verify_tree(dest);return read(dest)
    records={};paths=[OUT/'execution.json'];groups={};checks={}
    for key in KEYS:
        checked_runtime(key,p)
        rp=OUT/key/'runtime-validation.json';verify_tree(rp);paths.append(rp)
        ep=OUT/f'evaluation-{key}.json';verify_tree(ep);r=read(ep);validate_evaluation(r,key)
        records[key]=r;paths.append(ep)
    histpath=evaluator.HIST_EVAL/'completion.json';verify_tree(histpath);paths.append(histpath);hist=read(histpath)
    for arm in ('K','L'):
        name=f'{arm}-300';groups[name]=evaluator.aggregate([records[f'{name}-{s}'] for s in SEEDS])
        checks[name]=evaluator.policy_checks(groups[name],hist['aggregate']['R-300'],hist['historical_A'],p)
    extra=retention(groups['L-300'],groups['K-300'],p['retention']['tolerance'])
    checks['L-300']['checks'].extend(extra);checks['L-300']['passed']=all(x['passed'] for x in checks['L-300']['checks'])
    selected='L-300' if checks['L-300']['passed'] else None
    paired={str(s):paired_change(records[f'K-300-{s}']['rows'],records[f'L-300-{s}']['rows']) for s in SEEDS}
    return save(dest,dict(status='development_candidate_selected' if selected else 'complete_no_candidate',
        selected_family=selected,aggregate=groups,policy_results=checks,paired_K_to_L=paired,
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        completed_cells=list(KEYS),evaluation_role='viewed_development_not_blind_test',
        independent_scene_count=1,training_admitted=False,promotable=False,
        inputs={str(x):file_sha256(x) for x in paths}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--worker',choices=KEYS)
    args=ap.parse_args()
    if args.worker:worker(args.worker);return
    with (OUT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_execution();p=read(OUT/'protocol.json')
        if args.prepare_only:print('EXECUTION_READY',OUT,flush=True);return
        completed=[]
        for key in KEYS:
            verify_tree(OUT/'execution.json')
            rp=OUT/key/'runtime-validation.json'
            if rp.exists():verify_tree(rp);checked_runtime(key,p)
            else:run_worker(key)
            completed.append(key)
            save(OUT/'training-progress.json',dict(status='complete' if len(completed)==6 else 'in_progress',
                 protocol_identity=p['identity'],completed_cells=completed))
        print('FINAL',finalize(p)['status'],flush=True)

if __name__=='__main__':main()
