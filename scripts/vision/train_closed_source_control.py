"""Explicit training entry. Default is read-only and never starts training."""
import argparse,fcntl,os,signal,subprocess,sys,traceback
from pathlib import Path
from scripts.vision.freeze_material_retention_coverage import OUT,prior,freeze
from scripts.vision.finalize_material_retention_route import check_receipt
from scripts.vision.validate_compensated_loader_receipts import validate
from scripts.vision.train_frozen_multiscale import contract as reference_contract
from scripts.vision.frozen_multiscale_runtime import preprocess
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.preflight_closed_source_control import OUT,KEYS,protocol
from scripts.vision import train_material_retention_coverage as reference
def contract(key):
    if key not in KEYS:raise ValueError('Unknown same-source-dose cell')
    p=protocol();seed=key.split('-')[-1];old,_,_=reference.contract('T-'+seed)
    if p['initialization']!=old['initialization'] or p['evaluation']!=old['evaluation']:raise ValueError('Initialization/evaluation drift')
    if p['training_config'][key]!=old['training_config']['T-'+seed]:raise ValueError('Configuration drift')
    files=list((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
    if len(files)!=1:raise ValueError('Missing/duplicate actual preflight')
    r=prior.read(files[0]);prior.verify(r)
    check_actual(p,key,r['actual']);check_log(p,key,r['brightness_log'])
    if r['cell']!=key or len(r['batch_records'])!=450:raise ValueError('Incomplete preflight')
    return p,p,r

class BatchGate:
    def __init__(self,expected,lookup):self.expected=expected;self.lookup=lookup;self.step=0;self.actual=[];self.records=[]
    def apply(self,owner,batch):
        if self.step>=450:raise ValueError('Extra batch')
        e=self.expected['batch_records'][self.step];members=[self.lookup[x] for x in batch['im_file']]
        if members!=e['members'] or tensor_hash(batch['img'])!=e['image_tensor_sha256']:raise ValueError('Member/brightness/base tensor changed')
        for f,sha in e['full_supervision'].items():
            if tensor_hash(batch[f])!=sha:raise ValueError('Complete supervision changed')
        out=preprocess(owner,batch,640)
        if tuple(out['img'].shape)!=(6,3,640,640):raise ValueError('Effective training shape changed')
        for f,sha in e['full_supervision'].items():
            if tensor_hash(out[f])!=sha:raise ValueError('Effective supervision changed')
        self.actual.extend(members);self.records.append(e);self.step+=1;return out
    def finish(self,log):
        if self.step!=450 or self.actual!=self.expected['actual'] or log!=self.expected['brightness_log']:raise ValueError('Incomplete/changed actual exposure or brightness')

def ready():
    r=prior.read(OUT/'entry-ready.json');prior.verify(r)
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS):raise ValueError('Entry not ready')
    for key in KEYS:contract(key)

def complete(key):
    r=prior.read(OUT/'training'/key/'completion.json');prior.verify(r)
    p,s,e=contract(key)
    if r['cell']!=key or r['protocol_identity']!=p['identity'] or r['optimizer_steps']!=450 or r['checkpoint_selection']!='terminal_last_only':raise ValueError('Invalid training receipt')
    x=prior.read(r['exposure_path']);prior.verify(x)
    if x['actual']!=e['actual'] or x['batch_records']!=e['batch_records'] or x['brightness_log']!=e['brightness_log']:raise ValueError('Actual training inputs changed')
    if prior.file_sha256(r['weights'])!=r['weights_sha256']:raise ValueError('Changed endpoint')

def worker(key):
    ready();prior.verify(prior.read(OUT/'training-authorization.json'));p,source,expected=contract(key)
    root=OUT/'training'/key;root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():complete(key);return
    if any(prior.read(f)['status']=='semantic_failure' for f in root.glob('attempt-*/failure.json')):raise ValueError('Resolve semantic failure before retry')
    n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Training attempt budget exhausted')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        import torch,yaml
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer
        torch.set_num_threads(4);seed=int(key.rsplit('-',1)[1]);sk=key
        logs=[];gate=BatchGate(expected,{r['image_path']:r['member_id'] for r in source['pool_rows']})
        class Trainer(DetectionTrainer):
            step_count=0
            def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
                if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
                if dataset_path!=source['listings'][sk] or batch_size!=6 or max(int(self.model.stride.max()),32)!=32:raise ValueError('Unexpected loader configuration')
                return make_loader(make_dataset(source,sk,logs),source,sk,self)
            def preprocess_batch(self,batch):return gate.apply(self,batch)
            def optimizer_step(self):
                if self.step_count>=450:raise ValueError('Extra optimizer step')
                result=super().optimizer_step();self.step_count+=1;return result
        data=attempt/'dataset.yaml';data.write_text(yaml.safe_dump(dict(path=str(root),train=source['listings'][sk],val=source['listings'][sk],names=source['names'])))
        model=YOLO(source['initialization']['path'])
        model.train(trainer=Trainer,data=str(data),project=str(attempt),name='run',**source['training_config'][sk])
        gate.finish(logs);check_actual(source,sk,gate.actual);check_log(source,sk,logs)
        if model.trainer.step_count!=450:raise ValueError('Incorrect optimizer count')
        from scripts.vision.brightness_lr_retention import check_curve
        import csv
        args=attempt/'run/args.yaml';curve=attempt/'run/results.csv'
        check_curve(yaml.safe_load(args.read_text()),list(csv.DictReader(curve.open())),seed,.0005)
        weight=attempt/'run/weights/last.pt';xp=attempt/'exposure.json'
        prior.frozen(xp,dict(actual=gate.actual,batch_records=gate.records,brightness_log=logs,inputs={str(OUT/'entry-ready.json'):prior.file_sha256(OUT/'entry-ready.json')}))
        paths=[OUT/'entry-ready.json',OUT/'protocol.json',OUT/'training-authorization.json',Path(__file__).resolve(),xp,weight,data,args,curve]
        prior.frozen(root/'completion.json',dict(status='complete',cell=key,protocol_identity=p['identity'],optimizer_steps=450,
            weights=str(weight),weights_sha256=prior.file_sha256(weight),exposure_path=str(xp),checkpoint_selection='terminal_last_only',
            training_member_validation_role='training_fit_only_not_development',inputs={str(x):prior.file_sha256(x) for x in paths}))
        complete(key)
    except BaseException as ex:
        prior.frozen(attempt/'failure.json',dict(status='semantic_failure' if isinstance(ex,ValueError) else 'technical_or_cancelled_failure',error=traceback.format_exc()));raise

def cleanup(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def run():
    ready()
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=OUT/'training-authorization.json'
        if not auth.exists():prior.frozen(auth,dict(status='explicit_train_command_received',cells=list(KEYS),inputs={str(OUT/'entry-ready.json'):prior.file_sha256(OUT/'entry-ready.json')}))
        else:prior.verify(prior.read(auth))
        for key in KEYS:
            if (OUT/'training'/key/'completion.json').exists():complete(key);continue
            folder=OUT/'workers'/key;folder.mkdir(parents=True,exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
            if n>3:raise ValueError('Worker attempt cap')
            attempt=folder/f'attempt-{n:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'log.txt').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_closed_source_control','--train','--worker',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    print('STARTED',key,proc.pid,flush=True)
                    proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Worker failed; inspect '+str(attempt))
                complete(key)
            except BaseException:
                if proc is not None:cleanup(proc)
                prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
            finally:
                if proc is not None:cleanup(proc)
            print('TRAINED',key,flush=True)
        prior.frozen(OUT/'training/completion.json',dict(status='six_cells_trained_not_evaluated',cells=list(KEYS),inputs={str(OUT/'training'/k/'completion.json'):prior.file_sha256(OUT/'training'/k/'completion.json') for k in KEYS}))

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args(argv)
    if a.worker and not a.train:ap.error('--worker requires --train')
    if a.worker:worker(a.worker)
    elif a.train:run()
    else:ready();print('ENTRY_READY_NO_TRAINING_STARTED')

if __name__=='__main__':main()
