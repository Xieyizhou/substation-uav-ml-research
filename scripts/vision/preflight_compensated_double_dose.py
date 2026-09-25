"""Real paired loaders, full supervision and tensor logs; optimizer forbidden."""
import hashlib,traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.freeze_compensated_double_dose import OUT,prior,checks
from scripts.vision import train_compensated_material as low
from scripts.vision.train_frozen_multiscale import contract
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual

def tensor_hash(tensor):
    t=tensor.detach().cpu().contiguous()
    return hashlib.sha256((str(t.dtype)+str(tuple(t.shape))).encode()+t.numpy().tobytes()).hexdigest()

def run():
    import torch,ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);p=prior.read(OUT/'protocol.json');prior.verify(p)
    _,reference,_=contract('fixed-7')
    if {'torch':torch.__version__,'ultralytics':ultralytics.__version__}!=reference['environment']:raise ValueError('Environment changed')
    checks(prior.read(low.OUT/'protocol.json'),p)
    paths=[OUT/'protocol.json',Path(__file__).resolve(),Path(__import__('scripts.vision.brightness_transfer_runtime',fromlist=['x']).__file__),Path(__import__('scripts.vision.order_retention_runtime',fromlist=['x']).__file__)];lookup={x['image_path']:x for x in p['pool_rows']}
    for row in p['pool_rows']:
        for kind in ('image','label'):
            if prior.file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Pool file hash changed')
    if prior.file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Initialization changed')
    for seed in (7,17,27):
        root=OUT/'loader-checks'/f'seed-{seed}';root.mkdir(parents=True,exist_ok=True)
        complete=sorted(root.glob('attempt-*/complete.json'))
        if complete:
            if len(complete)!=1:raise ValueError('Ambiguous complete preflight')
            r=prior.read(complete[0]);prior.verify(r)
            for key,cell in r['cells'].items():check_actual(p,key,cell['actual']);check_log(p,key,cell['brightness_log'])
            paths.append(complete[0]);continue
        num=len(list(root.glob('attempt-*')))+1
        if num>3:raise ValueError('Preflight attempt cap exhausted')
        attempt=root/f'attempt-{num:02}';attempt.mkdir();keys=[f'V-{seed}',f'VM-{seed}']
        expected={k:low.contract(k)[2] for k in keys}
        logs={k:[] for k in keys};actual={k:[] for k in keys};batches={k:[] for k in keys};owner=SimpleNamespace(epoch=0)
        try:
            init_seeds(seed,deterministic=True)
            with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('Optimizer forbidden')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('Backward forbidden')),patch.object(torch.autograd,'backward',side_effect=AssertionError('Backward forbidden')),patch.object(YOLO,'train',side_effect=AssertionError('Training forbidden')),patch.object(YOLO,'val',side_effect=AssertionError('Validation forbidden')):
                loaders=[make_loader(make_dataset(p,k,logs[k]),p,k,owner) for k in keys]
                step=0
                for epoch in range(45):
                    owner.epoch=epoch
                    for a,b in zip(*loaders,strict=True):
                        for field in ('cls','bboxes','batch_idx'):
                            if not torch.equal(a[field],b[field]):raise ValueError('Paired complete supervision differs')
                        for j,(pa,pb) in enumerate(zip(a['im_file'],b['im_file'],strict=True)):
                            if lookup[pa].get('pair_id',lookup[pa]['member_id'])!=lookup[pb].get('pair_id',lookup[pb]['member_id']):raise ValueError('Paired source changed')
                            if pa==pb and not torch.equal(a['img'][j],b['img'][j]):raise ValueError('Untreated image tensor differs')
                        for key,batch in zip(keys,(a,b)):
                            if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Invalid batch shape')
                            members=[lookup[x]['member_id'] for x in batch['im_file']];actual[key].extend(members)
                            batches[key].append(dict(step=step,members=members,image_tensor_sha256=tensor_hash(batch['img']),
                                full_supervision={field:tensor_hash(batch[field]) for field in ('cls','bboxes','batch_idx')}))
                        step+=1
            if step!=450:raise ValueError('Incomplete actual optimization-step batches')
            for k in keys:check_actual(p,k,actual[k]);check_log(p,k,logs[k])
            for i,(a,b) in enumerate(zip(logs[keys[0]],logs[keys[1]],strict=True)):
                if a['gain']!=b['gain']:raise ValueError('Brightness differs across arms')
                if i not in p['replacement_positions'][str(seed)] and (a['before']!=b['before'] or a['after']!=b['after']):raise ValueError('Non-treatment augmentation changed')
            for k in keys:
                for oldlog,newlog in zip(expected[k]['brightness_log'],logs[k],strict=True):
                    if oldlog['gain']!=newlog['gain']:raise ValueError('Historical position brightness changed')
                    if oldlog['member_id']==newlog['member_id'] and (oldlog['before']!=newlog['before'] or oldlog['after']!=newlog['after']):raise ValueError('Unchanged member tensor drift')
            cells={k:dict(actual=actual[k],brightness_log=logs[k],batch_records=batches[k]) for k in keys}
            dest=attempt/'complete.json';prior.frozen(dest,dict(status='paired_actual_loaders_verified',cells=cells,seed=seed,
                optimizer_created=False,backward_executed=False,training_validation_run=False,
                inputs={str(x):prior.file_sha256(x) for x in [OUT/'protocol.json',Path(__file__).resolve()]+[Path(r[k+'_path']) for r in p['pool_rows'] for k in ('image','label')]}));paths.append(dest)
            print('PAIRED_PREFLIGHT_COMPLETE',seed,'5400 ACTUAL LOADS',flush=True)
        except BaseException:
            prior.frozen(attempt/'failure.json',dict(status='preflight_failed',reason=traceback.format_exc(),training_started=False,
                inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json')}));raise
    prior.frozen(OUT/'loader-completion.json',dict(status='six_loaders_verified_final_regression_pending',training_ready=False,training_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':run()

