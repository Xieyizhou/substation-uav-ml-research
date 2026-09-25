"""Identical real training engine for bounded CPU probes and full units."""
from pathlib import Path
import time,math
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.frozen_multiscale_runtime import preprocess
from scripts.vision.preflight_unified_lighting import tensor_hash

class ProbeFinished(Exception):pass

def loss_values(items):
    if isinstance(items,dict):
        return {str(k):float(v.detach().cpu()) for k,v in items.items()}
    return {str(i):float(v) for i,v in enumerate(items.detach().cpu())}

def execute(p,key,attempt,threads,expected,probe_steps=None):
    import torch,yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    logs=[];actual=[];records=[];steps=[];batch_started=[];lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    total=len(p['schedules'][key])//6
    class Trainer(DetectionTrainer):
        count=0
        def _setup_train(self):
            super()._setup_train()
            torch.set_num_threads(threads)
            if torch.get_num_threads()!=threads:raise ValueError('CPU thread setup failed')
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
            if dataset_path!=p['listings'][key] or batch_size!=6:raise ValueError('Loader configuration changed')
            return make_loader(make_dataset(p,key,logs),p,key,self)
        def preprocess_batch(self,batch):
            batch_started.append(time.monotonic())
            if self.count>=total or torch.get_num_threads()!=threads:raise ValueError('Training step/thread drift')
            e=expected['batch_records'][self.count];members=[lookup[x] for x in batch['im_file']]
            if members!=e['members'] or tensor_hash(batch['img'])!=e['image_tensor_sha256']:raise ValueError('Actual tensor differs from preflight')
            for f,sha in e['full_supervision'].items():
                if tensor_hash(batch[f])!=sha:raise ValueError('Full label tensor drift')
            out=preprocess(self,batch,640);actual.extend(members);records.append(e);return out
        def optimizer_step(self):
            super().optimizer_step();self.count+=1
            losses=loss_values(self.loss_items)
            if any(not math.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite training loss')
            steps.append(dict(step=self.count,loss_items=losses,seconds=time.monotonic()-batch_started[-1],threads=torch.get_num_threads()))
            if probe_steps and self.count==probe_steps:raise ProbeFinished()
    data=Path(attempt)/'dataset.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(attempt),train=p['listings'][key],val=p['listings'][key],names=p['names'])))
    model=YOLO(p['initialization']['path']);start=time.monotonic();probe_complete=False
    try:model.train(trainer=Trainer,data=str(data),project=str(attempt),name='run',**p['training_config'][key])
    except ProbeFinished:probe_complete=True
    target=probe_steps or total
    if len(steps)!=target or len(actual)!=target*6:raise ValueError('Incomplete optimization')
    if probe_steps:
        if not probe_complete:raise ValueError('Probe exceeded endpoint')
        if actual!=expected['actual'][:target*6] or logs!=expected['brightness_log'][:target*6]:raise ValueError('Probe input drift')
    else:
        check(p,key,actual,logs)
        if logs!=expected['brightness_log']:raise ValueError('Full brightness bytes changed')
    from src.ml.artifacts import object_sha256
    state_identity=object_sha256({n:tensor_hash(t) for n,t in model.trainer.model.state_dict().items()})
    return dict(actual=actual,batch_records=records,brightness_log=logs,optimizer_steps=len(steps),step_records=steps,
        wall_seconds=time.monotonic()-start,terminal_model_state_identity=state_identity,effective_threads=threads,
        probe_only=bool(probe_steps),training_admitted=False,promotable=False)
