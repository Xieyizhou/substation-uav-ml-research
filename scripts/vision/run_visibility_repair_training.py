"""Bounded isolated training; sampler gate runs before any optimizer step."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from scripts.vision.prepare_visibility_repair_training import OUT,ROOT,KEYS,prepare,read,save,file_sha256,verify_tree,baseline_verify
import scripts.vision.run_exposure_diagnosis as trainer
import scripts.vision.run_visibility_quality_training as evaluator
from scripts.vision.run_matched_appearance_training import validate_args,validate_evaluation

def verify_indices(expected,actual):
    if list(actual)!=list(expected):raise ValueError('Actual sampler index order differs from frozen plan')

def prepare_execution():
    p=prepare();ep=OUT/'execution.json'
    if ep.exists():verify_tree(ep);return p
    modules=['test_visibility_repair_design','test_visibility_repair_full_labels','test_visibility_repair_runtime','test_retained_bridge_review','test_canonical_shutdown']
    result=subprocess.run([sys.executable,'-m','unittest',*['tests.'+m for m in modules]],cwd=ROOT,capture_output=True,text=True,check=True)
    paths=[OUT/'protocol.json',Path(__file__),Path(trainer.__file__),Path(evaluator.__file__),ROOT/'scripts/vision/run_matched_appearance_training.py',*[ROOT/(m.replace('.','/')+'.py') for m in ['tests.'+x for x in modules]]]
    save(ep,dict(status='execution_preflight_passed_runtime_sampler_gate_still_required',regression_output=result.stderr,whole_repository_tested=False,cells=list(KEYS),baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),inputs={str(x):file_sha256(x) for x in paths}))
    return p

def worker(key):
    verify_tree(OUT/'execution.json');p=read(OUT/'protocol.json');trainer.OUT=OUT;evaluator.OUT=OUT
    if not (OUT/key/'completion.json').exists() and len(list((OUT/key).glob('attempt-*')))>=3:raise ValueError('Training attempt budget exhausted')
    import torch.utils.data
    import numpy as np
    import yaml
    OriginalLoader=torch.utils.data.DataLoader
    rows={r['member_id']:r for r in p['pool_rows']};inverse={r['image_path']:mid for mid,r in rows.items()}
    class CheckedLoader(OriginalLoader):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            sampler=kwargs.get('sampler')
            if not hasattr(sampler,'trainer') or not hasattr(sampler,'mapping'):return
            t=sampler.trainer;epoch=t.epoch;observed=[]
            try:
                for number in range(30):
                    t.epoch=number;observed.extend(inverse[self.dataset.im_files[i]] for i in self.sampler)
            finally:t.epoch=epoch
            verify_indices(p['schedules'][key],observed)
            for label in self.dataset.labels:
                mid=inverse[label['im_file']];text=Path(rows[mid]['label_path']).read_text().strip()
                expected=np.array([list(map(float,line.split())) for line in text.splitlines()],dtype=np.float32).reshape(-1,5)
                actual=np.concatenate((label['cls'],label['bboxes']),axis=1)
                if expected.shape!=actual.shape or not np.allclose(expected,actual,rtol=0,atol=1e-6):raise ValueError('Actual dataset full labels changed')
            save(Path(t.save_dir)/'sampler-preflight.json',dict(status='actual_sampler_and_full_labels_verified_before_optimization',cell=key,checked_draws=len(observed),checked_members=len(self.dataset.im_files),
                inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),str(Path(__file__)):file_sha256(Path(__file__))}))
            print('ACTUAL_SAMPLER_PREFLIGHT_PASSED',key,flush=True)
    torch.utils.data.DataLoader=CheckedLoader
    try:cell=trainer.train(key,{**p,'datasets':{key.split('-')[0]:p['datasets'][key]}})
    finally:torch.utils.data.DataLoader=OriginalLoader
    cell=trainer.checked_cell(OUT/key/'completion.json',p);folder=Path(cell['exposure_path']).parent
    verify_tree(folder/'sampler-preflight.json');validate_args(yaml.safe_load((folder/'args.yaml').read_text()))
    r=evaluator.evaluate(key,p);validate_evaluation(r,key)
    paths=[OUT/'execution.json',OUT/key/'completion.json',folder/'sampler-preflight.json',OUT/f'evaluation-{key}.json']
    save(OUT/key/'runtime-validation.json',dict(status='training_and_evaluation_complete_prediction_review_pending',cell=key,inputs={str(x):file_sha256(x) for x in paths}))
    print('CELL_COMPLETE_EVALUATED',key,r['negative_summary'],flush=True)

def run_worker(key):
    root=OUT/'worker-attempts'/key;root.mkdir(parents=True,exist_ok=True);number=len(list(root.glob('attempt-*')))+1
    if number>3:raise ValueError('Worker attempt budget exhausted')
    attempt=root/f'attempt-{number:03}';attempt.mkdir();proc=None
    try:
        with (attempt/'worker.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.run_visibility_repair_training','--worker',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic()
            while proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('RUNNING',key,round(time.monotonic()-start),'seconds',flush=True)
                    if time.monotonic()-start>21600:raise TimeoutError('Six-hour cell limit')
            if proc.returncode:raise RuntimeError(f'Worker failed {key}; see {attempt}/worker.log')
        save(attempt/'result.json',dict(status='complete',cell=key,inputs={str(attempt/'worker.log'):file_sha256(attempt/'worker.log')}))
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        save(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=KEYS);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if args.worker:worker(args.worker);return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare_execution()
        if args.prepare_only:print('EXECUTION_READY');return
        completed=[]
        for key in KEYS:
            verify_tree(OUT/'execution.json');rp=OUT/key/'runtime-validation.json'
            if rp.exists():verify_tree(rp)
            else:run_worker(key)
            completed.append(key);save(OUT/'progress.json',dict(status='all_six_evaluated_pending_explicit_review' if len(completed)==6 else 'in_progress',completed_cells=completed,inputs={str(OUT/k/'runtime-validation.json'):file_sha256(OUT/k/'runtime-validation.json') for k in completed}))
        print('ALL_SIX_DONE_REVIEW_REQUIRED',flush=True)

if __name__=='__main__':main()
