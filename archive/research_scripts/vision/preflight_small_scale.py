"""Six real loaders, 39,600 draws; optimization/backward/validation forbidden."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.small_scale_control import OUT, REF, KEYS, freeze, prior
from scripts.vision.closed_budget_runtime import make_dataset, make_loader, check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden


def preflight():
    p = freeze(); dest = OUT/'loader-feasibility.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    deps = [OUT/'design.json', Path(__file__).resolve()]
    deps += [Path(__file__).with_name(n+'.py') for n in ('closed_budget_runtime', 'order_retention_runtime', 'brightness_transfer_runtime', 'closed_budget_engine_v2')]
    lookup = {r['image_path']:r['member_id'] for r in p['pool_rows']}
    units = []; reference_batches = {}
    for key in KEYS:
        seed = int(key.split('-')[-1])
        cp = REF/'training'/f'ICG1000-{seed}'/'completion.json'
        c = prior.read(cp); prior.verify(c)
        xp = Path(c['exposure_path']); old = prior.read(xp); prior.verify(old)
        deps += [cp, xp]
        logs, actual, batches = [], [], []; owner = SimpleNamespace(epoch=0)
        init_seeds(seed, deterministic=True)
        with ExitStack() as stack:
            for obj, name in ((torch.optim.Optimizer, '__init__'), (torch.Tensor, 'backward'),
                              (torch.autograd, 'backward'), (YOLO, 'train'), (YOLO, 'val')):
                stack.enter_context(patch.object(obj, name, forbidden))
            loader = make_loader(make_dataset(p, key, logs), p, key, owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch = epoch
                for batch in loader:
                    i = len(batches); members = [lookup[x] for x in batch['im_file']]
                    sha = tensor_hash(batch['img'])
                    labels = {f:tensor_hash(batch[f]) for f in ('cls', 'bboxes', 'batch_idx')}
                    if tuple(batch['img'].shape) != (6,3,640,640) or members != p['schedules'][key][6*i:6*i+6]:
                        raise ValueError('Actual member/shape drift')
                    if i < 1000:
                        previous = old['batch_records'][i]
                        if labels != previous['full_supervision'] or sha != previous['image_tensor_sha256']:
                            raise ValueError('Historical prefix tensor drift')
                    if key.startswith('SM') and labels != reference_batches[seed][i]['full_supervision']:
                        raise ValueError('Paired full-supervision tensor drift')
                    actual.extend(members)
                    batches.append(dict(step=i, members=members, image_tensor_sha256=sha, full_supervision=labels))
        check(p, key, actual, logs)
        if len(batches) != 1100: raise ValueError('Incomplete loader')
        if logs[:6000] != old['brightness_log']: raise ValueError('Historical augmentation prefix drift')
        if key.startswith('SR'): reference_batches[seed] = batches
        units.append(dict(cell=key, actual=actual, brightness_log=logs, batch_records=batches))
        print('ACTUAL_PREFLIGHT_COMPLETE', key, len(actual), flush=True)
    return prior.frozen(dest, dict(status='six_actual_loaders_verified_no_training', units=units,
        optimizer_created=False, backward_executed=False, validation_run=False,
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(preflight()['status'])
