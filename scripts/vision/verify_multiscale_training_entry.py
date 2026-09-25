"""Exercise the installed training hook without training or optimizer creation."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import subprocess,sys
from scripts.vision.train_frozen_multiscale import OUT,KEYS,contract,BatchGate,prior
from scripts.vision.brightness_transfer_runtime import make_dataset
from scripts.vision.order_retention_runtime import make_loader
from scripts.vision.preflight_frozen_multiscale import forbidden
from scripts.vision.verify_experiment_baseline import verify as baseline_verify

def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);paths=[OUT/'protocol.json',OUT/'completion.json',Path(__file__).resolve(),prior.ROOT/'scripts/vision/train_frozen_multiscale.py'];checked=[]
    for key in KEYS:
        p,source,e=contract(key);seed=int(key.rsplit('-',1)[1]);sk=f'R-clean-{seed}';log=[]
        owner=SimpleNamespace(epoch=0,device=torch.device('cpu'),args=SimpleNamespace(multi_scale=0),stride=32)
        gate=BatchGate(e,{r['image_path']:r['member_id'] for r in source['pool_rows']})
        init_seeds(seed,deterministic=True)
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(source,sk,log),source,sk,owner)
            for batch in loader:gate.apply(owner,batch)
        if gate.step!=10 or log!=e['brightness_log'][:60] or gate.actual!=e['actual'][:60]:raise ValueError('Training hook prefix mismatch')
        checked.append(dict(cell=key,actual_hook_batches=10,actual_hook_loads=60,sizes=[b['size'] for b in gate.records],all_hook_tensors_equal_preflight=True))
        print('HOOK_VERIFIED',key,'10 real batches; no optimizer',flush=True)
    if any(set(c['sizes'])!={320,640,960} for c in checked if c['cell'].startswith('multiscale')):raise ValueError('Smoke did not cover all scales')
    suites=['tests.test_multiscale_training_entry','tests.test_frozen_multiscale_runtime','tests.test_material_transfer_controls','tests.test_brightness_transfer','tests.test_order_retention']
    test=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=60)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'entry-ready.json',dict(status='training_entry_verified_not_started',cells=list(KEYS),hook_smoke=checked,
        prior_full_preflight=dict(cells=6,loads=16200,batches=2700),
        training_started=False,training_authorized=False,optimizer_created=False,backward_executed=False,
        reference_policy='Explicit --train runs six independent cells; does not silently reuse historical weights or switch accelerator.',
        tests_output=test.stdout+test.stderr,baseline=baseline,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(test.stdout+test.stderr);print('ENTRY_VERIFIED_NO_TRAINING')

if __name__=='__main__':main()
