"""Frozen, step-matched development experiment: uniform vs hierarchical sampling.

Only the already-used 30 diagnostic training frames are eligible. Labels remain
complete, including incidental objects. No held/protected labels enter sampling.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from src.vision.collection.simulator_labels import instance_simulator_label

BASE = ROOT / 'data/research/ml_training_recovery_v1'
OUT = BASE / 'stratified-sampling-v1'
NAMES = ['transformer', 'switchgear', 'capacitor_bank', 'reactor']
SEEDS = [7, 17, 27]
ARMS = ['uniform', 'stratified']
EPOCHS, SLOTS, BATCH = 10, 30, 6
WEIGHTS = ROOT / 'models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt'


def area_bucket(box):
    a, b, c, d = box
    area = (c - a) * (d - b)
    return 'small' if area < 40000 else 'medium' if area < 180000 else 'large'


def expected_object(row):
    label = instance_simulator_label(row['expected_category'], row['expected_object_id'])
    matches = [o for o in row['truth']['objects'] if o['class_name'] == row['expected_category']
               and int(re.search(r'instance-(\d+)-', o['annotation_id'])[1]) == label]
    if len(matches) != 1:
        raise ValueError(f'Expected instance not uniquely identifiable: {row["view_id"]}')
    return matches[0]


def make_schedule(rows, arm, seed):
    """Balance expected class, then available maps, then available scale bins.

    Exhausted strata cycle over their original images: draws are exposures, never
    extra unique examples. Empty strata are not filled with invented samples.
    """
    rng = random.Random(seed)
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, row in enumerate(rows):
        tree[row['expected_category']][row['map_id']][row['scale']].append(i)
    counts = Counter()

    def take(options, prefix):
        options = sorted(options)
        minimum = min(counts[prefix + (x,)] for x in options)
        value = rng.choice([x for x in options if counts[prefix + (x,)] == minimum])
        counts[prefix + (value,)] += 1
        return value

    epochs = []
    for epoch in range(EPOCHS):
        if arm == 'uniform':
            selected = list(range(len(rows)))
        elif arm == 'stratified':
            selected = []
            for _ in range(SLOTS):
                cls = take(tree, ('class',))
                map_id = take(tree[cls], ('map', cls))
                scale = take(tree[cls][map_id], ('scale', cls, map_id))
                selected.append(take(tree[cls][map_id][scale], ('image', cls, map_id, scale)))
        else:
            raise ValueError(arm)
        rng.shuffle(selected)
        if len(selected) != SLOTS:
            raise ValueError('Each epoch must have exactly 30 draws')
        epochs.append(selected)
    return epochs


def exposures(rows, schedule):
    draws = [i for epoch in schedule for i in epoch]
    return {
        'draws': len(draws), 'unique_frames': len(set(draws)),
        'by_expected_class': dict(Counter(rows[i]['expected_category'] for i in draws)),
        'by_map': dict(Counter(rows[i]['map_id'] for i in draws)),
        'by_source': dict(Counter(rows[i]['source'] for i in draws)),
        'by_stratum': dict(Counter(':'.join((rows[i]['expected_category'], rows[i]['map_id'], rows[i]['scale'])) for i in draws)),
        'all_object_exposures': dict(Counter(o['class_name'] for i in draws for o in rows[i]['truth']['objects'])),
        'per_frame_draws': {rows[i]['view_id']: count for i, count in Counter(draws).items()},
    }


def prepare():
    if (OUT / 'protocol.json').exists():
        protocol = read_record(OUT / 'protocol.json')
        for path, digest in protocol['inputs'].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f'Frozen input changed: {path}')
        return protocol
    mem_path = BASE / 'memorization-v1/protocol.json'
    receipt_path = BASE / 'simple-annotation-repair-v2/capture/collection-receipt.json'
    review_path = BASE / 'simple-annotation-repair-v2/semantic-review.json'
    mem, receipt, review = map(read_record, [mem_path, receipt_path, review_path])
    accepted = {r['view_id'] for r in review['frames'] if r['semantic_decision'] == 'accepted'}
    if set(r['view_id'] for r in receipt['views']) != accepted:
        raise ValueError('All repaired training frames must have a semantic review')
    inputs = {str(p): file_sha256(p) for p in [mem_path, receipt_path, review_path, WEIGHTS, Path(__file__)]}
    if inputs[str(WEIGHTS)] != '820f882a5d375be0a294555c7168d49b6d68c21adce97f00b45fe7b0ebc54836':
        raise ValueError('Baseline weights mismatch')
    rows = []
    for source, members in [('original', [i['row'] for i in mem['selected']]), ('repaired_simple', receipt['views'])]:
        for r in members:
            src = Path(r['rgb_path'])
            inputs[str(src)] = file_sha256(src)
            if inputs[str(src)] != r['image_sha256']:
                raise ValueError('Image mismatch')
            try:
                anchor = expected_object(r)
                anchor_status = 'planned_instance_present'
            except ValueError:
                # Preserve the common training pool but never claim an incidental
                # same-class object is the originally planned instance.
                candidates = [o for o in r['truth']['objects'] if o['class_name'] == r['expected_category']]
                if not candidates:
                    raise ValueError('No observed class available for sampling')
                anchor = max(candidates, key=lambda o: (o['bbox_xyxy'][2]-o['bbox_xyxy'][0])*(o['bbox_xyxy'][3]-o['bbox_xyxy'][1]))
                anchor_status = 'planned_instance_absent_use_observed_class_for_sampling_only'
            rows.append({**r, 'source': source, 'scale': area_bucket(anchor['bbox_xyxy']),
                         'sampling_anchor_annotation_id': anchor['annotation_id'], 'sampling_anchor_status': anchor_status})
    if len(rows) != 30 or len({r['image_sha256'] for r in rows}) != 30:
        raise ValueError('Expected exactly 30 unique training images')
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'labels'):
        (OUT / name).mkdir()
    for i, row in enumerate(rows):
        image_path = OUT / 'images' / f'{i:03}.png'
        with Image.open(row['rgb_path']) as image:
            if image.size != (1920, 1080):
                raise ValueError('Unexpected image size')
            image.convert('RGB').save(image_path)
        lines = []
        for obj in row['truth']['objects']:
            a, b, c, d = obj['bbox_xyxy']
            if not (0 <= a < c <= 1920 and 0 <= b < d <= 1080):
                raise ValueError('Invalid box')
            lines.append(f'{NAMES.index(obj["class_name"])} {(a+c)/3840:.10f} {(b+d)/2160:.10f} {(c-a)/1920:.10f} {(d-b)/1080:.10f}')
        label_path = OUT / 'labels' / f'{i:03}.txt'
        label_path.write_text('\n'.join(lines) + '\n')
        row['training_path'] = str(image_path)
        inputs.update({str(p): file_sha256(p) for p in (image_path, label_path)})
    dataset = OUT / 'dataset.yaml'
    dataset.write_text(f'path: {OUT}\ntrain: images\nval: images\nnames: {json.dumps(NAMES)}\n')
    inputs[str(dataset)] = file_sha256(dataset)
    schedules = {f'{arm}-{seed}': make_schedule(rows, arm, seed) for seed in SEEDS for arm in ARMS}
    strata = {':'.join((r['expected_category'], r['map_id'], r['scale'])) for r in rows}
    protocol = {
        'status': 'frozen_before_training', 'rows': rows, 'inputs': inputs, 'schedules': schedules,
        'exposures': {key: exposures(rows, value) for key, value in schedules.items()},
        'missing_strata': [f'{c}:{m}:{s}' for c in NAMES for m in ('simple', 'medium', 'complex')
                           for s in ('small', 'medium', 'large') if f'{c}:{m}:{s}' not in strata],
        'epochs': EPOCHS, 'batch': BATCH, 'nbs': BATCH, 'optimizer_steps_per_run': 50,
        'seeds': SEEDS, 'arms': ARMS, 'weights': str(WEIGHTS), 'dataset': str(dataset),
        'comparison_rule': 'Paired seed summaries on hash-disjoint development controls and development replay; no best-seed or threshold selection. A sampling recipe is not accepted if mean instance hits or expected-class hits decrease in either of these groups.',
        'training_admitted': False, 'promotable': False,
        'limits': ['Repeated exposure does not create class/scene/scale diversity.',
                   'Anchor-class balancing retains every incidental object label; object counts need not be balanced.',
                   'Three paired seeds and reused correlated development images cannot establish generalization.',
                   'Automatic trainer validation uses training images only. Always evaluate last.pt.'],
    }
    protocol['identity'] = object_sha256(protocol)
    write_json(OUT / 'protocol.json', protocol)
    return protocol


def train(arm, seed):
    from torch.utils.data import DataLoader, Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import torch
    import ultralytics

    protocol = prepare()
    key = f'{arm}-{seed}'
    schedule = protocol['schedules'][key]
    completion = OUT / key / 'completion.json'
    if completion.exists():
        result = read_record(completion)
        if file_sha256(Path(result['weights'])) != result['weights_sha256']:
            raise ValueError('Completed run weights changed')
        return result
    if (OUT / key).exists():
        raise ValueError(f'Incomplete run exists; inspect before retry: {key}')
    paths = [r['training_path'] for r in protocol['rows']]
    actual = []

    class ScheduledSampler(Sampler):
        def __init__(self, trainer, mapping):
            self.trainer, self.mapping = trainer, mapping

        def __len__(self):
            return SLOTS

        def __iter__(self):
            return iter(self.mapping[paths[i]] for i in schedule[self.trainer.epoch])

    class Trainer(DetectionTrainer):
        step_count = 0

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode != 'train':
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            mapping = {path: i for i, path in enumerate(dataset.im_files)}
            if set(mapping) != set(paths) or batch_size != BATCH:
                raise ValueError('Unexpected training membership/batch size')
            return DataLoader(dataset, batch_size=BATCH, sampler=ScheduledSampler(self, mapping),
                              num_workers=0, collate_fn=dataset.collate_fn,
                              generator=torch.Generator().manual_seed(seed))

        def preprocess_batch(self, batch):
            actual.append({'epoch': self.epoch, 'paths': list(batch['im_file'])})
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            self.step_count += 1
            return super().optimizer_step()

    model = YOLO(str(WEIGHTS))
    model.train(trainer=Trainer, data=protocol['dataset'], epochs=EPOCHS, imgsz=640,
                batch=BATCH, nbs=BATCH, device='cpu', workers=0, optimizer='AdamW',
                lr0=.001, lrf=1, warmup_epochs=0, warmup_bias_lr=0, seed=seed,
                deterministic=True, patience=0, amp=False, mosaic=0, close_mosaic=0,
                mixup=0, copy_paste=0, degrees=0, translate=0, scale=0, shear=0,
                perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0,
                project=str(OUT), name=key, plots=False, save=True, val=False)
    if model.trainer.step_count != 50 or len(actual) != 50:
        raise ValueError('Training steps did not match the protocol')
    for epoch in range(EPOCHS):
        observed = [p for b in actual if b['epoch'] == epoch for p in b['paths']]
        if observed != [paths[i] for i in schedule[epoch]]:
            raise ValueError('Observed sampling differs from frozen schedule')
    weights = OUT / key / 'weights/last.pt'
    result = {'protocol_identity': protocol['identity'], 'arm': arm, 'seed': seed,
              'status': 'complete', 'optimizer_steps': model.trainer.step_count,
              'observed_batches': actual, 'weights': str(weights), 'weights_sha256': file_sha256(weights),
              'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
              'training_admitted': False, 'promotable': False}
    result['identity'] = object_sha256(result)
    write_json(completion, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.arm is None:
        p = prepare()
        print(json.dumps({'identity': p['identity'], 'exposures': p['exposures'], 'missing_strata': p['missing_strata']}, indent=2))
    elif args.seed is None:
        parser.error('--arm requires --seed')
    else:
        result = train(args.arm, args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != 'observed_batches'}, indent=2))
