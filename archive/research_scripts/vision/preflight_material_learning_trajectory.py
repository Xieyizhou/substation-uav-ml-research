"""Verify observation and historical real-loader receipts without training."""
import copy
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.train_material_learning_trajectory import OUT,KEYS,prior,contract,freeze_trajectory
from scripts.vision.material_trajectory_runtime import capture
from scripts.vision.evaluate_scale_endpoints import baseline_verify
from scripts.vision.infer_material_member_fit import forbidden


def main():
    p=freeze_trajectory();dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    paths=[OUT/'protocol.json',Path(__file__).resolve()]
    checks={}
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),
                         (torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
            stack.enter_context(patch.object(obj,name,forbidden))
        for key in KEYS:
            source,_,expected=contract(key)
            if len(expected['actual'])!=2700 or len(expected['batch_records'])!=450:
                raise ValueError('Historical real-loader coverage missing')
            model=YOLO(source['initialization']['path']).model.cpu().float()
            owner=SimpleNamespace(model=model,ema=SimpleNamespace(ema=copy.deepcopy(model),updates=0),
                                  args=SimpleNamespace(**source['training_config'][key]))
            with tempfile.TemporaryDirectory(prefix='trajectory-preflight-') as d:
                record=capture(owner,Path(d)/'probe.pt',0)
                loaded=torch.load(record['path'],map_location='cpu',weights_only=False)
                from scripts.vision.material_trajectory_runtime import state_digest
                if state_digest(loaded['model'])!=record['raw_state_sha256']:raise ValueError('Reload drift')
                size=Path(record['path']).stat().st_size
            checks[key]=dict(historical_actual_samples_verified=2700,batches_verified=450,
                            capture_and_reload_passed=True,estimated_checkpoint_bytes=size*len(p['cells'][key]['checkpoint_steps']))
    needed=sum(x['estimated_checkpoint_bytes'] for x in checks.values())+2_000_000_000
    free=shutil.disk_usage(OUT).free
    if free<needed:raise ValueError(f'Insufficient disk: need {needed}, free {free}')
    suites=['tests.test_material_learning_trajectory','tests.test_material_trajectory_runtime',
            'tests.test_material_retention_entry','tests.test_material_retention_coverage',
            'tests.test_compensated_loader_receipts']
    r=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=120)
    if r.returncode:raise ValueError(r.stdout+r.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths.extend(prior.ROOT/'scripts/vision'/f for f in ('train_material_learning_trajectory.py','material_trajectory_runtime.py'))
    paths.extend(prior.ROOT/(s.replace('.','/')+'.py') for s in suites)
    return prior.frozen(dest,dict(status='trajectory_entry_verified_not_started',protocol_identity=p['identity'],
        checks=checks,baseline=baseline,tests=r.stdout+r.stderr,disk_free=free,estimated_required=needed,
        optimizer_created=False,backward_executed=False,training_validation_executed=False,
        loader_reuse='Verified historical identical actual sequences, full supervision and brightness receipts; not a new traversal.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
