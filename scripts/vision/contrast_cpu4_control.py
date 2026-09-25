"""Matched fresh four-thread reference/contrast arms, automatic bounded sequence."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision import contrast_transfer_control as base
from scripts.vision.locked_cpu_threads import locked_threads
from scripts.vision import locked_cpu_threads as lock_module

ROOT=base.OUT.parent/'contrast-transfer-cpu4-control-v2'
ABORTED=base.OUT
KEYS=tuple(f'cpu4-transfer-480-{s}' for s in (7,17,27))
ARM='reference'
OUT=ROOT/ARM
checked=base.checked

def validate(p,source,arm):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if p[f]!=source[f]:raise ValueError('Unexpected non-contrast change '+f)
    expected=copy.deepcopy(source['configuration']);expected['augmentation']='off' if arm=='reference' else base.VERSION
    if p['configuration']!=expected:raise ValueError('Configuration drift')
    for seed,key,old in zip((7,17,27),KEYS,base.previous.KEYS):
        for f in ('schedules','exposures','windows','listings'):
            if p[f][key]!=source[f][old]:raise ValueError('Exposure drift')
        values=[1.]*2880 if arm=='reference' else base.coefficients(seed)
        if p['contrast'][key]!=values:raise ValueError('Coefficient drift')

def freeze():
    old=checked(base.SOURCE/'protocol.json');path=OUT/'protocol.json'
    if path.exists():
        p=checked(path);validate(p,old,ARM);return p
    OUT.mkdir(parents=True,exist_ok=True)
    parent=checked(ABORTED/'protocol.json');p=copy.deepcopy(parent)
    for f in ('inputs','identity'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings'):p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,base.previous.KEYS)}
    p['contrast']={k:([1.]*2880 if ARM=='reference' else base.coefficients(s)) for k,s in zip(KEYS,(7,17,27))}
    p['configuration']['augmentation']='off' if ARM=='reference' else base.VERSION
    p.update(status='cpu4_paired_preflight_pending',arm=ARM,
        design='Two fresh matched arms at actual enforced four Torch intra-op threads, three seeds per arm. Same initialization, LR, members, order, labels and 480 steps. Only contrast differs between arms.',
        thread_erratum='Historical trainer calls set_num_threads(4) before select_device(cpu), which resets to library NUM_THREADS=8. The v1 contrast attempt was interrupted and is excluded. Historical weights are context, not the matched CPU4 reference.',
        direct_control=str(ROOT/'reference'),training_admitted=False,promotable=False)
    if ARM=='reference':p['contrast_definition']='Identity transform: coefficient 1 for all 2880 draws; uint8 tensors must remain identical to historical raw loader output.'
    deps=dict(parent['inputs'])
    for f in (ABORTED/'protocol.json',Path(__file__),Path(lock_module.__file__),Path(base.__file__)):
        deps[str(f.resolve())]=file_sha256(f)
    if ARM=='contrast':
        for key in KEYS:
            for name in ('completion.json','tensor-verification.json','thread-verification.json'):
                f=ROOT/'reference/training'/key/name;checked(f);deps[str(f.resolve())]=file_sha256(f)
    p['inputs']=deps;validate(p,old,ARM);return write_record(path,p)

def worker(key):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from unittest.mock import patch
    original=DetectionTrainer.optimizer_step;steps=[]
    def optimizer_step(self,*args,**kwargs):
        n=torch.get_num_threads()
        if n!=4:raise RuntimeError('Actual optimizer step thread count differs')
        result=original(self,*args,**kwargs);steps.append(n);return result
    with locked_threads(4) as events,patch.object(DetectionTrainer,'optimizer_step',optimizer_step):result=base.worker(key)
    tp=OUT/'training'/key/'thread-verification.json'
    if tp.exists():checked(tp)
    else:
        if len(steps)!=480:raise ValueError('Missing runtime thread observations')
        write_record(tp,dict(status='all_480_optimizer_steps_actual_threads_verified',threads=4,optimizer_step_threads=steps,setter_events=events,
            training_admitted=False,promotable=False,inputs={str(f.resolve()):file_sha256(f) for f in (OUT/'training'/key/'completion.json',Path(__file__),Path(lock_module.__file__))}))
    return result

def summarize():
    runtime=base.runtime;records=[];comparisons=[];deps={};queue=[]
    for seed,key,old_key in zip((7,17,27),KEYS,base.previous.KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        ref=base.SOURCE if ARM=='reference' else ROOT/'reference';refkey=old_key if ARM=='reference' else key
        op=ref/'evaluation-v1/units'/refkey/'completion.json';o=checked(op);old=checked(o['result']);by={(x['pair_id'],x['variant']):x for x in old['rows']}
        for row in r['rows']:comparisons.append(dict(seed=seed,pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for f in (cp,op,Path(c['result']),Path(o['result']),OUT/'training'/key/'thread-verification.json'):checked(f);deps[str(f.resolve())]=file_sha256(f)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_retention_gates_pending',arm=ARM,
        comparison_kind='historical_execution_context_only' if ARM=='reference' else 'matched_cpu4_contrast_only',group=runtime.evaluation.aggregate(records),instance_comparisons=comparisons,
        negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure(arm):
    global ARM,OUT
    ARM=arm;OUT=ROOT/arm;base.OUT=OUT;base.KEYS=KEYS
    runtime=base.runtime;runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=base.loader
    runtime.MODULE='scripts.vision.contrast_cpu4_control'
    # Each child is launched with its arm in an environment variable, never guessed from artifacts.
    os.environ['CONTRAST_CONTROL_ARM']=arm
    runtime.TESTS=runtime.TESTS+('tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')
    runtime.summarize=summarize

def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    for arm in ('reference','contrast'):
        proc=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--arm',arm,'--train'],start_new_session=True)
        try:
            proc.wait(timeout=14400)
            if proc.returncode:raise RuntimeError('Arm failed: '+arm)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('reference','contrast'),default=os.environ.get('CONTRAST_CONTROL_ARM','reference'))
    ap.add_argument('--run',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args()
    if a.run:run()
    else:
        configure(a.arm)
        if a.worker:worker(a.worker)
        elif a.eval_worker:base.runtime.eval_worker(a.eval_worker)
        elif a.train:base.runtime.train()
        else:base.runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
