"""Execute independent cells; preserve failed attempts and verify resumed artifacts."""
import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT, exposures, file_sha256, prepare, read, save, verify


def checked_cell(path, protocol):
    row = read(path)
    verify(row)
    if row['protocol_identity'] != protocol['identity'] or row['status'] != 'complete':
        raise ValueError('Cell protocol or status invalid')
    key = row['cell']
    actual = read(row['exposure_path'])
    verify(actual)
    if actual['draws'] != protocol['schedules'][key] or actual['summary'] != protocol['exposures'][key]:
        raise ValueError('Actual exposure mismatch')
    if row['optimizer_steps'] * 6 != len(actual['draws']):
        raise ValueError('Optimizer step mismatch')
    return row


def train(key, protocol):
    import torch
    import ultralytics
    from torch.utils.data import DataLoader, Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    arm, steps, seed = key.split('-')
    steps, seed = int(steps), int(seed)
    cell = OUT / key
    completion = cell / 'completion.json'
    if completion.exists():
        return checked_cell(completion, protocol)
    cell.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (cell / f'attempt-{attempt:03}').exists():
        attempt += 1
    name = f'attempt-{attempt:03}'
    rows = {r['member_id']: r for r in protocol['pool_rows']}
    member_for_path = {r['image_path']: r['member_id'] for r in rows.values()}
    expected = protocol['schedules'][key]
    actual = []

    class Schedule(Sampler):
        def __init__(self, trainer, mapping):
            self.trainer, self.mapping = trainer, mapping
        def __len__(self):
            return 60
        def __iter__(self):
            start = self.trainer.epoch * 60
            return iter(self.mapping[rows[mid]['image_path']] for mid in expected[start:start + 60])

    class Trainer(DetectionTrainer):
        step_count = 0
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode != 'train':
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            mapping = {p: i for i, p in enumerate(dataset.im_files)}
            if set(mapping) != {rows[mid]['image_path'] for mid in expected} or batch_size != 6:
                raise ValueError('Runtime membership changed')
            return DataLoader(dataset, batch_size=6, sampler=Schedule(self, mapping), num_workers=0,
                              collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(seed))
        def preprocess_batch(self, batch):
            actual.extend(member_for_path[p] for p in batch['im_file'])
            return super().preprocess_batch(batch)
        def optimizer_step(self):
            self.step_count += 1
            return super().optimizer_step()
    try:
        model = YOLO(protocol['controls']['initial_weights'])
        model.train(trainer=Trainer, data=protocol['datasets'][arm], epochs=steps // 10,
                    imgsz=640, batch=6, nbs=6, device='cpu', workers=0, optimizer='AdamW',
                    lr0=.001, lrf=1., warmup_epochs=0, warmup_bias_lr=0, seed=seed,
                    deterministic=True, patience=0, amp=False, mosaic=0, close_mosaic=0,
                    mixup=0, copy_paste=0, degrees=0, translate=0, scale=0, shear=0,
                    perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0,
                    project=str(cell), name=name, plots=False, save=True, val=False)
        if actual != expected or model.trainer.step_count != steps:
            raise ValueError('Actual exposure or optimizer steps differ')
        run = cell / name
        exposure_path = run / 'exposure.json'
        save(exposure_path, {'draws': actual, 'summary': exposures(protocol['pool_rows'], actual)})
        weight = run / 'weights/last.pt'
        inputs = {str(p): file_sha256(p) for p in (weight, exposure_path, run / 'results.csv', run / 'args.yaml', Path(__file__))}
        result = save(completion, {'status': 'complete', 'cell': key, 'protocol_identity': protocol['identity'],
            'weights': str(weight), 'weights_sha256': file_sha256(weight), 'exposure_path': str(exposure_path),
            'optimizer_steps': steps, 'exposure_verified': True, 'inputs': inputs,
            'ultralytics_version': ultralytics.__version__, 'torch_version': torch.__version__})
        print(f'CELL_COMPLETE {key} {result["identity"]}', flush=True)
        return result
    except BaseException:
        save(cell / name / 'failure.json', {'status': 'failed', 'cell': key, 'error': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell')
    args = parser.parse_args()
    protocol = prepare()
    keys = [args.cell] if args.cell else [f'{arm}-{steps}-{seed}' for steps in (100, 300) for arm in 'RXY' for seed in (7, 17, 27)]
    completed = []
    for key in keys:
        verify(protocol)
        completed.append(train(key, protocol))
        save(OUT / 'training-progress.json', {'status': 'complete' if len(completed) == 18 else 'in_progress',
             'protocol_identity': protocol['identity'], 'completed_cells': [r['cell'] for r in completed]})


if __name__ == '__main__':
    main()
