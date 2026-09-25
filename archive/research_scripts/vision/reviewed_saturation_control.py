"""Independent saturation-only experiment; default is optimizer-free preflight.

The explicit --train chain includes fixed development inference, never automatic
semantic approval. Historical files and held-frame decisions remain untouched.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision import reviewed_negative_order_control as source
from scripts.vision import evaluate_reviewed_negative_order as evaluation
from scripts.vision import train_reactor_visibility_expansion as trainer

OUT = source.OUT.parent / 'reviewed-saturation-control-v1'
SEEDS = (7, 17, 27)
KEYS = tuple(f'saturation-480-{s}' for s in SEEDS)
MODULE = 'scripts.vision.reviewed_saturation_control'
OBSERVED = {}


def checked(path):
    return evaluation.checked(path)


def factors(seed):
    positions = sorted(range(2880), key=lambda i: hashlib.sha256(f'saturation-v1|{seed}|{i}'.encode()).hexdigest())
    result = [None] * 2880
    for j, i in enumerate(positions):
        result[i] = (0., .5, 1.)[j % 3]
    return result


def desaturate(images, values):
    import torch
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3 or len(values) != len(images):
        raise ValueError('Expected uint8 RGB BCHW and one frozen factor per image')
    if any(v not in (0., .5, 1.) for v in values):
        raise ValueError('Unexpected saturation factor')
    x = images.float()
    gray = x[:, 0:1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114
    a = torch.tensor(values, dtype=torch.float32, device=x.device).reshape(-1, 1, 1, 1)
    return (gray + a * (x - gray)).round().clamp(0, 255).to(torch.uint8)


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_protocol(p, old):
    if p['pool_rows'] != old['pool_rows'] or p['initialization'] != old['initialization']:
        raise ValueError('Member/initialization drift')
    for seed, key in zip(SEEDS, KEYS):
        ref = f'reviewed-interleaved-480-{seed}'
        if p['schedules'][key] != old['schedules'][ref] or p['exposures'][key] != old['exposures'][ref]:
            raise ValueError('Exposure drift')
        if p['saturation'][key] != factors(seed) or Counter(p['saturation'][key]) != {0.: 960, .5: 960, 1.: 960}:
            raise ValueError('Frozen factor drift')
    for row in p['pool_rows']:
        for kind in ('image', 'label'):
            if file_sha256(row[kind + '_path']) != row[kind + '_sha256']:
                raise ValueError('Member changed')


def freeze():
    old = checked(source.OUT / 'protocol.json')
    checked(source.REVIEW)
    import torch, ultralytics, platform
    current_environment=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,platform=platform.platform())
    if old['environment'] != current_environment: raise ValueError('Historical control environment differs')
    dest = OUT / 'protocol.json'
    if dest.exists():
        p = checked(dest); validate_protocol(p, old); return p
    OUT.mkdir(parents=True, exist_ok=True)
    fit = source.OUT / 'training-fit-diagnosis-v1'
    deps = [source.OUT / 'protocol.json', source.REVIEW, fit / 'summary.json', fit / 'negative-coverage.json',
            source.OUT / 'evaluation-v1/error-review-v1/review.json', Path(__file__), Path(trainer.__file__),
            Path(evaluation.__file__),Path(source.__file__),Path(evaluation.ev.__file__)]
    from scripts.vision import order_retention_runtime
    deps.append(Path(order_retention_runtime.__file__))
    for item in deps[:5]: checked(item)
    observations = [
        dict(indices=[0, 15, 51], observation='Gray body with dark rectangular front, beside slender poles; body truncated by right image edge.'),
        dict(indices=[13, 14, 24, 37, 53, 64, 73, 74, 109], observation='Gray rectangular body/dark front behind or beside teal foreground body and vertical poles; overlap is represented.'),
        dict(indices=[43, 61, 79, 80, 102, 112], observation='Gray body dark rectangular face and base are visible; views and foreground pole overlap vary.'),
        dict(indices=[36, 42, 50, 66, 70, 81, 103, 107, 110, 117], observation='Gray block top/side/back and base visible at varying scale; front-panel evidence is not asserted.')]
    coverage = checked(fit / 'negative-coverage.json')
    review_path = OUT / 'coverage-screening.json'
    if not review_path.exists():
        write_record(review_path, dict(status='visual_coverage_screening_not_training_admission', review_nature='AI辅助审核',
            reviewed_at=datetime.now(timezone.utc).isoformat(), observations=observations,
            inspected_indices=list(range(120)), members=coverage['members'],
            limits='Four contact pages inspected. Named positive evidence establishes broad shape presence, not exhaustive ROI quality, pixel visibility, asset identity or sufficient independent coverage. No new member approval. Existing whole-frame holds remain.',
            training_admitted=False, promotable=False,
            inputs={str(x.resolve()): file_sha256(x) for x in [fit / 'negative-coverage.json'] + [fit / f'negative-coverage-{i}.png' for i in range(4)]}))
    deps.append(review_path)
    p = {k: old[k] for k in ('pool_rows', 'names', 'initialization', 'evaluation', 'environment', 'held_members')}
    p.update(schedules={}, exposures={}, listings={}, saturation={}, windows={})
    for seed, key in zip(SEEDS, KEYS):
        ref = f'reviewed-interleaved-480-{seed}'
        for field in ('schedules', 'exposures', 'listings', 'windows'): p[field][key] = old[field][ref]
        p['saturation'][key] = factors(seed)
        cpath = source.OUT / 'training' / ref / 'completion.json'; c = checked(cpath)
        e = checked(c['exposure_path'])
        if e['actual'] != p['schedules'][key] or c['optimizer_steps'] != 480:
            raise ValueError('Historical control not compatible')
        deps.extend([cpath, Path(c['exposure_path']), Path(c['weights']), Path(p['listings'][key])])
    p.update(status='frozen_preflight_pending', training_config={**old['training_config'],'enhancement':'frozen_saturation_only'},
        configuration=dict(steps=480, batch=6, nbs=6, lr=.0005, optimizer='AdamW', cpu_threads_per_worker=4,
            max_parallel_workers=2, initialization='independent_v2.11', checkpoint='last.pt', augmentation='frozen_saturation_only'),
        intervention='RGB saturation toward BT.601 luma, factors 0/0.5/1 each 960 draws, hash assigned independently of label, no spatial transform. All other augmentation off.',
        comparison='Three new seeds versus valid historical reviewed-interleaved seeds with identical source images, labels, ordered batches, exposures, optimizer and budget. Only saturation intervention differs.',
        interpretation='Tests one color robustness intervention, not a proof that color uniquely causes errors. Luma retained approximately after uint8 rounding; geometry and all labels unchanged. Development set has been viewed; no new independent scenes.',
        acceptance='Reuse all fixed development and retention gates; no thresholds/seed selection changes. Numerical inference is not semantic review or candidate approval.',
        automatic_followup='After all three verified endpoints, run both fixed confidence protocols on 48 paired and 48 negative development images, generate pending error-review queue. Never auto-approve.',
        training_admitted=False, promotable=False, inputs={str(x.resolve()): file_sha256(x) for x in deps})
    validate_protocol(p, old)
    return write_record(dest, p)


def loader(p, key, owner):
    dataset, raw = source.loader(p, key, owner)
    reference_path = OUT / 'actual-preflight' / f'{key}.json'
    expected = checked(reference_path)['tensor_records'] if reference_path.exists() else None
    records = OBSERVED.setdefault(key, [])
    class Wrapped:
        def __len__(self): return len(raw)
        def __getattr__(self, name): return getattr(raw, name)
        def __iter__(self):
            for j, batch in enumerate(raw):
                position = owner.epoch * 60 + j * 6
                values = p['saturation'][key][position:position+6]
                before = {k: tensor_hash(batch[k]) for k in ('cls', 'bboxes', 'batch_idx')}
                original = tensor_hash(batch['img'])
                batch['img'] = desaturate(batch['img'], values)
                record = dict(position=position, factors=values, original_tensor_sha256=original,
                    augmented_tensor_sha256=tensor_hash(batch['img']), label_tensors=before)
                if before != {k: tensor_hash(batch[k]) for k in before}: raise ValueError('Augmentation changed labels')
                if expected is not None and record != expected[position // 6]: raise ValueError('Runtime tensor differs from preflight')
                records.append(record)
                yield batch
    return dataset, Wrapped()


def preflight():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    p = freeze(); root = OUT / 'actual-preflight'; root.mkdir(exist_ok=True)
    inverse = {r['image_path']: r['member_id'] for r in p['pool_rows']}
    for key in KEYS:
        path = root / f'{key}.json'
        if path.exists():
            r = checked(path)
        else:
            owner = SimpleNamespace(epoch=0); actual=[]; OBSERVED[key]=[]
            with patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('Optimizer forbidden')), patch.object(torch.Tensor, 'backward', side_effect=AssertionError('Backward forbidden')), patch.object(YOLO, 'train', side_effect=AssertionError('Training forbidden')), patch.object(YOLO, 'val', side_effect=AssertionError('Validation forbidden')):
                dataset, batches = loader(p, key, owner)
                for epoch in range(48):
                    owner.epoch=epoch
                    for batch in batches:
                        if tuple(batch['img'].shape) != (6, 3, 640, 640): raise ValueError('Shape drift')
                        actual.extend(inverse[x] for x in batch['im_file'])
            if actual != p['schedules'][key] or len(OBSERVED[key]) != 480: raise ValueError('Preflight exposure drift')
            r=write_record(path, dict(status='actual_complete_loader_verified_no_training', key=key, actual=actual,
                tensor_records=OBSERVED[key], optimizer_created=False, backward_executed=False, validation_run=False,
                training_admitted=False, promotable=False, inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))
        if r['actual'] != p['schedules'][key] or len(r['tensor_records']) != 480: raise ValueError('Invalid preflight')
        print('PREFLIGHT_COMPLETE', key, 2880, flush=True)
    path=OUT/'entry-ready.json'
    if not path.exists():
        write_record(path, dict(status='ready_for_training_not_started',cells=list(KEYS),actual_draws_verified=8640,
            optimizer_created=False, backward_executed=False,training_admitted=False,promotable=False,
            inputs={str(x.resolve()):file_sha256(x) for x in [OUT/'protocol.json']+[root/f'{k}.json' for k in KEYS]}))
    return p, checked(path)


def worker(key):
    p, ready = preflight(); OBSERVED[key]=[]
    trainer.OUT=OUT.resolve();trainer.KEYS=KEYS;trainer.STEPS=480;trainer._loader=loader
    trainer._ready=lambda: (p, ready)
    done=OUT/'training'/key/'completion.json'
    if done.exists():
        c=checked(done);e=checked(c['exposure_path']);a=checked(OUT/'training'/key/'augmentation.json')
        if c['optimizer_steps']!=480 or e['actual']!=p['schedules'][key] or a['records']!=checked(OUT/'actual-preflight'/f'{key}.json')['tensor_records']:
            raise ValueError('Reusable unit drift')
        return c
    c=trainer._worker(key)
    expected=checked(OUT/'actual-preflight'/f'{key}.json')['tensor_records']
    if OBSERVED[key] != expected: raise ValueError('Actual augmentation sequence incomplete')
    write_record(OUT/'training'/key/'augmentation.json',dict(status='actual_augmentation_matches_preflight',records=OBSERVED[key],
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in [done,OUT/'actual-preflight'/f'{key}.json']}))
    return c


def parallel(flag, keys, folder):
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(keys),2):
        jobs=[]
        try:
            for key in keys[start:start+2]:
                n=len(list(folder.glob(f'{key}-attempt-*.log')))+1
                if n>3:raise ValueError('Technical attempt cap')
                log=(folder/f'{key}-attempt-{n:03}.log').open('x')
                proc=subprocess.Popen([sys.executable,'-u','-m',MODULE,flag,key],stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log,key));print('STARTED',flag,key,proc.pid,flush=True)
            for proc,_,key in jobs:
                proc.wait(timeout=14400)
                if proc.returncode:raise RuntimeError('Unit failed '+key)
        finally:
            for proc,log,_ in jobs:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()


def eval_worker(key):
    evaluation.TRAIN=OUT;evaluation.OUT=OUT/'evaluation-v1';evaluation.KEYS=KEYS
    return evaluation.worker(key)


def summarize():
    records=[];deps={};comparisons=[];queue=[]
    for seed,key in zip(SEEDS,KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        ref=checked(source.OUT/'evaluation-v1/units'/f'reviewed-interleaved-480-{seed}'/'completion.json');old=checked(ref['result'])
        for x,y in zip(old['rows'],r['rows']): comparisons.append(dict(seed=seed,pair_id=x['pair_id'],variant=x['variant'],instances=evaluation.compare_truth(x,y)))
        for row in r['negative_rows']:
            for i,pred in enumerate(row['predictions']):queue.append(dict(seed=seed,image_sha256=row['image_sha256'],view_id=row['view_id'],prediction_index=i,prediction=pred,review_status='pending'))
        for path in (cp,Path(c['result']),Path(ref['result'])):deps[str(path.resolve())]=file_sha256(path)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_retention_gates_pending',
        group=evaluation.aggregate(records),instance_comparisons=comparisons,negative_fp_review_queue=queue,
        matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,
        training_admitted=False,promotable=False,inputs=deps))


def train():
    preflight();root=OUT/'runner';root.mkdir(exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parallel('--worker',KEYS,root)
        paths=[OUT/'training'/k/name for k in KEYS for name in ('completion.json','augmentation.json')]
        for path in paths:checked(path)
        cp=root/'completion.json'
        if not cp.exists():write_record(cp,dict(status='three_units_complete_evaluation_queued',training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths}))
        parallel('--eval-worker',KEYS,OUT/'evaluation-v1')
        if not (OUT/'evaluation-v1/summary.json').exists():summarize()
        print('TRAINING_AND_NUMERICAL_EVALUATION_COMPLETE_REVIEW_PENDING',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--train',action='store_true')
    parser.add_argument('--worker',choices=KEYS);parser.add_argument('--eval-worker',choices=KEYS);a=parser.parse_args()
    if a.worker:worker(a.worker)
    elif a.eval_worker:eval_worker(a.eval_worker)
    elif a.train:train()
    elif a.freeze:print(freeze()['status'])
    else:preflight()
