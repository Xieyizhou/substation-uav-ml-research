"""Explicit training only; default is a read-only readiness check."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.exposure_order_retention import OUT,KEYS,ROOT,read,save,file_sha256,verify_tree
from scripts.vision.preflight_order_retention import validate_ready
from scripts.vision.order_retention_runtime import make_dataset,make_loader,overrides,check_actual

def worker(key):
    p=validate_ready()
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import scripts.vision.run_exposure_diagnosis as old
    import scripts.vision.run_visibility_quality_training as evaluator
    from scripts.vision.run_matched_appearance_training import validate_evaluation
    cellroot=OUT/'training'/key;cellroot.mkdir(parents=True,exist_ok=True)
    cp=cellroot/'completion.json'
    old.OUT=OUT/'training';evaluator.OUT=OUT/'training'
    if cp.exists():
        old.checked_cell(cp,p)
    else:
        count=len(list(cellroot.glob('attempt-*')))
        if count>=3:raise ValueError('Training attempt limit')
        name=f'attempt-{count+1:03}';actual=[];inverse={r['image_path']:r['member_id'] for r in p['pool_rows']}
        class Trainer(DetectionTrainer):
            step_count=0
            def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
                if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
                if batch_size!=6 or dataset_path!=p['listings'][key]:raise ValueError('Runtime loader config mismatch')
                if max(int(self.model.stride.max()),32)!=32:raise ValueError('Stride differs from preflight')
                return make_loader(make_dataset(p,key),p,key,self)
            def preprocess_batch(self,batch):
                actual.extend(inverse[x] for x in batch['im_file']);return super().preprocess_batch(batch)
            def optimizer_step(self):
                self.step_count+=1;return super().optimizer_step()
        try:
            model=YOLO(p['initialization']['path'])
            model.train(trainer=Trainer,data=p['datasets'][key],project=str(cellroot),name=name,**overrides(int(key.split('-')[-1])))
            check_actual(p,key,actual)
            if model.trainer.step_count!=450:raise ValueError('Wrong optimizer count')
            folder=cellroot/name;ep=folder/'exposure.json'
            save(ep,dict(draws=actual,summary=p['exposures'][key]));weight=folder/'weights/last.pt'
            paths=[ep,weight,folder/'args.yaml',folder/'results.csv',OUT/'ready.json',Path(__file__)]
            save(cp,dict(status='complete',cell=key,protocol_identity=p['identity'],weights=str(weight),weights_sha256=file_sha256(weight),
                exposure_path=str(ep),optimizer_steps=450,exposure_verified=True,training_member_validation_role='fitting_diagnostic_only',
                inputs={str(x):file_sha256(x) for x in paths}))
        except BaseException:
            save(cellroot/name/'failure.json',dict(status='failed',error=traceback.format_exc()));raise
    r=evaluator.evaluate(key,p);validate_evaluation(r,key)
    save(cellroot/'evaluated.json',dict(status='complete_pending_explicit_review',inputs={str(cp):file_sha256(cp),str(OUT/'training'/f'evaluation-{key}.json'):file_sha256(OUT/'training'/f'evaluation-{key}.json')}))

def isolated(key):
    root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True)
    count=len(list(root.glob('attempt-*')))
    if count>=3:raise ValueError('Worker attempt limit')
    attempt=root/f'attempt-{count+1:03}';attempt.mkdir();proc=None
    try:
        with (attempt/'log.txt').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_order_retention','--train','--worker',key],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            proc.wait(timeout=21600)
            if proc.returncode:raise RuntimeError(f'Worker failed: {key}')
        save(attempt/'result.json',dict(status='complete',inputs={str(attempt/'log.txt'):file_sha256(attempt/'log.txt')}))
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        save(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);args=ap.parse_args()
    if args.worker and not args.train:ap.error('--worker requires --train')
    if not args.train:validate_ready();print('READY_NO_TRAINING_STARTED');return
    if args.worker:worker(args.worker);return
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);validate_ready()
        for key in KEYS:
            done=OUT/'training'/key/'evaluated.json'
            if done.exists():verify_tree(done)
            else:isolated(key)
            print('EVALUATED',key,flush=True)

if __name__=='__main__':main()
