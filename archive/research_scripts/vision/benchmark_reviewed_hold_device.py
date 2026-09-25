"""Explicit short device benchmark, not a new scientific training family."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from scripts.vision.train_reviewed_hold import OUT as SOURCE, prior, overrides

OUT=SOURCE.parent.parent/'device-acceleration-benchmark-v1'/'runtime-fix-v3'
KEY='brightness-450-7'
CELLS=('cpu-1','mps-1','mps-2','cpu-2')

def check_prefix(p,draws,log,expected):
    if draws!=p['schedules'][KEY][:180] or log!=expected[:180]:
        raise ValueError('Benchmark sampling or brightness drift')

def freeze():
    import torch, ultralytics
    cp=SOURCE.parent/'completion.json'; pp=SOURCE/'protocol.json'
    for path in (cp,pp,SOURCE/'preflight'/f'{KEY}.json'):prior.verify(prior.read(path))
    p=prior.read(pp)
    if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    if prior.file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Initialization changed')
    paths=[cp,pp,SOURCE/'preflight'/f'{KEY}.json',Path(__file__).resolve(),Path(p['initialization']['path'])]
    paths += [prior.ROOT/'scripts/vision'/x for x in ('brightness_transfer_runtime.py','order_retention_runtime.py')]
    rec=dict(status='frozen_benchmark_only',cells=list(CELLS),steps=30,warmup_steps=10,seed=7,
        environment=dict(platform=platform.platform(),torch=torch.__version__,ultralytics=ultralytics.__version__),
        config={**overrides(7),'epochs':3,'save':False},
        differences_from_scientific_run=['30 rather than 450 steps','validation and checkpoint writes disabled','device only differs between benchmark cells'],
        timing='synchronized preprocessing-through-optimizer, excludes loader; total train wall time also reported',
        precision='float32 amp=False; MPS fallback forbidden; no numerical equivalence assumed',
        inputs={str(x):prior.file_sha256(x) for x in paths})
    dest=OUT/'protocol.json';OUT.mkdir(exist_ok=True)
    if dest.exists():
        r=prior.read(dest);prior.verify(r)
        if r['environment']!=rec['environment']:raise ValueError('Environment drift')
        return r
    return prior.frozen(dest,rec)

def worker(cell,attempt):
    import torch,yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from scripts.vision.brightness_transfer_runtime import make_dataset
    from scripts.vision.order_retention_runtime import make_loader
    torch.set_num_threads(4)
    f=freeze();p=prior.read(SOURCE/'protocol.json');device=cell.split('-')[0]
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise ValueError('Fallback not explicitly disabled')
    log=[];draws=[];rows=[];idx={r['image_path']:r['member_id'] for r in p['pool_rows']}
    def sync():
        if device=='mps':torch.mps.synchronize()
    def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    class Trainer(DetectionTrainer):
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
            return make_loader(make_dataset(p,KEY,log),p,KEY,self)
        def preprocess_batch(self,b):
            draws.extend(idx[x] for x in b['im_file'])
            self.batch_hash={k:digest(b[k]) for k in ('img','cls','bboxes','batch_idx')}
            sync();self.started=time.perf_counter()
            return super().preprocess_batch(b)
        def optimizer_step(self):
            super().optimizer_step();sync();elapsed=time.perf_counter()-self.started
            loss={k:float(v.detach().cpu()) for k,v in self.loss_items.items()}
            if not all(__import__('math').isfinite(x) for x in loss.values()):raise ValueError('Nonfinite loss')
            rows.append(dict(step=len(rows)+1,seconds=elapsed,loss=loss,batch_hash=self.batch_hash))
        def validate(self):return self.metrics,0.0
        def final_eval(self):pass
        def save_model(self):pass
    cfg=attempt/'dataset.yaml'
    cfg.write_text(yaml.safe_dump(dict(path=str(SOURCE/'export'),train=p['listings'][KEY],val=p['listings'][KEY],names=p['names'])))
    y=YOLO(p['initialization']['path']);start=time.perf_counter()
    # Use the same real trainer directly: YOLO.train insists on reloading a saved
    # checkpoint after training, whereas this disposable benchmark exports none.
    y.trainer=Trainer(overrides={**y.overrides,**f['config'],'device':device,
        'data':str(cfg),'project':str(attempt),'name':'run','mode':'train',
        'model':p['initialization']['path']},_callbacks=y.callbacks)
    y.trainer.model=y.trainer.get_model(weights=y.model,cfg=y.model.yaml)
    y.trainer.train()
    sync();wall=time.perf_counter()-start
    if len(rows)!=30:raise ValueError('Wrong step count')
    check_prefix(p,draws,log,prior.read(SOURCE/'preflight'/f'{KEY}.json')['actual_brightness'])
    # Disposable model state is summarized only; no candidate checkpoint is exported.
    state={k:digest(v) for k,v in y.trainer.model.state_dict().items()}
    paths=[OUT/'protocol.json',cfg,attempt/'run/args.yaml',attempt/'run/results.csv']
    prior.frozen(attempt/'completion.json',dict(status='short_benchmark_complete',cell=cell,steps=rows,
        draw_count=len(draws),draws=draws,brightness=log,state_hashes=state,wall_seconds=wall,
        median_step_seconds=statistics.median(r['seconds'] for r in rows[10:]),
        validation_run=False,candidate_checkpoint_exported=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))

def run():
    freeze()
    completed=[]
    for cell in CELLS:
        root=OUT/cell;root.mkdir(exist_ok=True)
        valid=list(root.glob('attempt-*/completion.json'))
        if valid:
            if len(valid)!=1:raise ValueError('Ambiguous complete cell')
            prior.verify(prior.read(valid[0]));completed.append(valid[0]);continue
        n=len(list(root.glob('attempt-*')))+1
        if n>3:raise ValueError('Technical attempt cap')
        attempt=root/f'attempt-{n:03}';attempt.mkdir();proc=None
        try:
            with (attempt/'log.txt').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--worker',cell,'--attempt',str(attempt)],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0'})
                proc.wait(timeout=900)
                if proc.returncode:raise RuntimeError('Benchmark failed: '+str(attempt/'log.txt'))
            completed.append(attempt/'completion.json')
        except BaseException:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
        print('BENCHMARK_COMPLETE',cell,flush=True)
    rs=[prior.read(x) for x in completed]
    for r in rs:
        prior.verify(r)
        if [s['batch_hash'] for s in r['steps']]!=[s['batch_hash'] for s in rs[0]['steps']]:raise ValueError('Backend input tensors differ')
    med={d:statistics.median(r['median_step_seconds'] for r in rs if r['cell'].startswith(d)) for d in ('cpu','mps')}
    prior.frozen(OUT/'summary.json',dict(status='benchmark_complete_not_scientific_equivalence',median_step_seconds=med,
        speedup_cpu_over_mps=med['cpu']/med['mps'],batch_tensors_identical=True,
        cells=[dict(cell=r['cell'],median=r['median_step_seconds'],wall=r['wall_seconds'],loss_first=r['steps'][0]['loss'],loss_last=r['steps'][-1]['loss']) for r in rs],
        inputs={str(x):prior.file_sha256(x) for x in completed}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--worker',choices=CELLS);ap.add_argument('--attempt',type=Path);a=ap.parse_args()
    if a.worker:
        if a.attempt is None:ap.error('--attempt required')
        worker(a.worker,a.attempt)
    elif a.run:run()
    else:freeze();print('PREFLIGHT_ONLY_NO_OPTIMIZER')
