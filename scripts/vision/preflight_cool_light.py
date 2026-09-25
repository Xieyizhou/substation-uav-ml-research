"""Actual 18,000 draws; optimization and validation explicitly forbidden."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.cool_light_control import OUT, REF, KEYS, freeze, prior
from scripts.vision.closed_budget_runtime import make_dataset, make_loader, check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden


def preflight():
    p = freeze()
    dest = OUT/'loader-feasibility.json'
    if dest.exists():
        r = prior.read(dest)
        prior.verify(r)
        return r
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    deps = [OUT/'design.json', Path(__file__).resolve()]
    for name in ('closed_budget_runtime.py', 'order_retention_runtime.py', 'brightness_transfer_runtime.py', 'closed_budget_engine_v2.py'):
        deps.append(Path(__file__).with_name(name))
    lookup = {r['image_path']: r['member_id'] for r in p['pool_rows']}
    units = []
    for key in KEYS:
        seed = key.split('-')[-1]
        cp = REF/'training'/f'IL1000-{seed}'/'completion.json'
        c = prior.read(cp)
        prior.verify(c)
        xp = Path(c['exposure_path'])
        old = prior.read(xp)
        prior.verify(old)
        deps.extend([cp, xp])
        logs, actual, batches = [], [], []
        owner = SimpleNamespace(epoch=0)
        init_seeds(int(seed), deterministic=True)
        with ExitStack() as stack:
            for obj, name in ((torch.optim.Optimizer, '__init__'), (torch.Tensor, 'backward'), (torch.autograd, 'backward'), (YOLO, 'train'), (YOLO, 'val')):
                stack.enter_context(patch.object(obj, name, forbidden))
            loader = make_loader(make_dataset(p, key, logs), p, key, owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch = epoch
                for batch in loader:
                    i = len(batches)
                    members = [lookup[x] for x in batch['im_file']]
                    sha = tensor_hash(batch['img'])
                    labels = {f: tensor_hash(batch[f]) for f in ('cls', 'bboxes', 'batch_idx')}
                    previous = old['batch_records'][i]
                    if tuple(batch['img'].shape) != (6, 3, 640, 640) or members != p['schedules'][key][i*6:i*6+6]:
                        raise ValueError('Actual loader member/shape drift')
                    if labels != previous['full_supervision']:
                        raise ValueError('Actual full label tensor changed')
                    if members == previous['members'] and sha != previous['image_tensor_sha256']:
                        raise ValueError('Unchanged batch image tensor drift')
                    actual.extend(members)
                    batches.append(dict(step=i, members=members, image_tensor_sha256=sha, full_supervision=labels))
        check(p, key, actual, logs)
        if len(batches) != 1000:
            raise ValueError('Incomplete actual loader')
        for i, (a, b) in enumerate(zip(logs, old['brightness_log'], strict=True)):
            if a['gain'] != b['gain'] or a['position'] != b['position']:
                raise ValueError('Actual brightness changed')
            if actual[i] == old['actual'][i] and a != b:
                raise ValueError('Unchanged augmentation bytes drift')
        units.append(dict(cell=key, actual=actual, brightness_log=logs, batch_records=batches))
        print('ACTUAL_PREFLIGHT_COMPLETE', key, len(actual), flush=True)
    return prior.frozen(dest, dict(status='three_actual_loaders_verified_no_training', units=units,
        optimizer_created=False, backward_executed=False, validation_run=False,
        inputs={str(d): prior.file_sha256(d) for d in deps}))


if __name__ == '__main__':
    print(preflight()['status'])
