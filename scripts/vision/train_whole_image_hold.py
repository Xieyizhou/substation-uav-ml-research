"""Explicit six-cell training; immutable ready receipt and common loader required."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.prepare_whole_image_hold import OUT, ROOT, read, verify, frozen, file_sha256
from scripts.vision.order_retention_runtime import make_dataset, make_loader, overrides, check_actual
from scripts.vision.finalize_whole_image_hold import validate_units

KEYS=tuple(f'{arm}-450-{seed}' for seed in (7,17,27) for arm in ('reference','hold'))

def ready():
    r=read(OUT/'ready.json');verify(r);p=read(OUT/'protocol.json');verify(p)
    if r['status']!='ready_for_training_not_started' or set(r['cells'])!=set(KEYS):raise ValueError('Readiness missing')
    validate_units(p,{k:read(OUT/'preflight'/k/'completion.json') for k in KEYS})
    if file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Stale initialization')
    return p

def complete(path,p,key):
    c=read(path);verify(c)
    if c['status']!='complete' or c['protocol_identity']!=p['identity'] or c['cell']!=key or c['optimizer_steps']!=450:raise ValueError('Invalid training receipt')
    e=read(c['exposure_path']);verify(e);check_actual(p,key,e['draws'])
    if file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Stale terminal weights')

def worker(key):
    p=ready();root=OUT/'training'/key;root.mkdir(parents=True,exist_ok=True);done=root/'completion.json'
    if done.exists():complete(done,p,key);return
    attempts=list(root.glob('attempt-*'))
    if len(attempts)>=3:raise ValueError('Training attempt budget exhausted')
    attempt=root/f'attempt-{len(attempts)+1:03}';attempt.mkdir()
    try:
        import yaml
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer
        config=attempt/'dataset.yaml'
        config.write_text(yaml.safe_dump(dict(path=str(OUT/'export'),train=p['listings'][key],val=p['listings'][key],names=p['names'])))
        actual=[];inverse={r['image_path']:r['member_id'] for r in p['pool_rows']}
        class Trainer(DetectionTrainer):
            step_count=0
            def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
                if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
                if dataset_path!=p['listings'][key] or batch_size!=6:raise ValueError('Loader configuration changed')
                if max(int(self.model.stride.max()),32)!=32:raise ValueError('Unexpected stride')
                return make_loader(make_dataset(p,key),p,key,self)
            def preprocess_batch(self,batch):
                actual.extend(inverse[x] for x in batch['im_file']);return super().preprocess_batch(batch)
            def optimizer_step(self):
                self.step_count+=1;return super().optimizer_step()
        model=YOLO(p['initialization']['path'])
        model.train(trainer=Trainer,data=str(config),project=str(attempt),name='run',**overrides(int(key.split('-')[-1])))
        check_actual(p,key,actual)
        if model.trainer.step_count!=450:raise ValueError('Incorrect optimizer steps')
        folder=attempt/'run';weight=folder/'weights/last.pt';ep=attempt/'exposure.json'
        frozen(ep,dict(draws=actual,summary=p['ledger'][key]['total'],windows_50_steps=p['ledger'][key]['windows_50_steps'],
            inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}))
        paths=[OUT/'ready.json',OUT/'protocol.json',Path(__file__),ep,weight,config,folder/'args.yaml',folder/'results.csv']
        frozen(done,dict(status='complete',cell=key,protocol_identity=p['identity'],weights=str(weight),weights_sha256=file_sha256(weight),
            exposure_path=str(ep),optimizer_steps=450,exposure_verified=True,checkpoint_selection='terminal_last_only',
            training_member_validation_role='training_fitting_diagnostic_only_not_development_evaluation',
            inputs={str(x):file_sha256(x) for x in paths}))
    except BaseException:
        frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc()));raise

def isolated(key):
    root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True);attempts=list(root.glob('attempt-*'))
    if len(attempts)>=3:raise ValueError('Worker budget exhausted')
    attempt=root/f'attempt-{len(attempts)+1:03}';attempt.mkdir();proc=None
    try:
        with (attempt/'log.txt').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_whole_image_hold','--train','--worker',key],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            frozen(attempt/'launch.json',dict(pid=proc.pid,cell=key,inputs={str(Path(__file__)):file_sha256(Path(__file__)),str(OUT/'ready.json'):file_sha256(OUT/'ready.json')}))
            proc.wait(timeout=21600)
            if proc.returncode:raise RuntimeError(f'Worker failed: {key}; inspect {attempt}')
        frozen(attempt/'completion.json',dict(status='complete',process_cleanup_confirmed=True,inputs={str(attempt/'log.txt'):file_sha256(attempt/'log.txt')}))
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args(argv)
    if a.worker and not a.train:ap.error('--worker requires --train')
    if not a.train:ready();print('READY_NO_TRAINING_STARTED');return
    if a.worker:worker(a.worker);return
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=ready()
        for key in KEYS:
            done=OUT/'training'/key/'completion.json'
            if done.exists():complete(done,p,key)
            else:isolated(key);complete(done,p,key)
            print('TRAINED',key,flush=True)
        dest=OUT/'training/completion.json'
        if not dest.exists():
            paths=[OUT/'training'/key/'completion.json' for key in KEYS]
            frozen(dest,dict(status='all_six_trained_evaluation_not_run',cells=list(KEYS),inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':main()
