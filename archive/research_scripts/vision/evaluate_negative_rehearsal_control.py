"""Explicit fixed-development evaluation and paired comparison of rehearsal."""
import argparse
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from src.vision.canonical.plan import read_record, write_record
from src.ml.artifacts import file_sha256
from scripts.vision import evaluate_reactor_visibility_expansion as ev
from scripts.vision.prepare_negative_rehearsal_control import OUT, SOURCE, KEYS


def verified(path):
    r = read_record(path)
    for p, h in r.get('inputs', {}).items():
        if file_sha256(p) != h:
            raise ValueError('Stale dependency: ' + p)
    return r


def run():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    ev.TRAIN = OUT
    ev.OUT = OUT / 'evaluation-v1'
    ev.OUT.mkdir(exist_ok=True)
    paired, negatives, inputs = ev._load_inputs()
    inputs[str(Path(__file__).resolve())] = file_sha256(Path(__file__).resolve())
    p = verified(OUT / 'protocol.json')
    inputs[str(OUT / 'protocol.json')] = file_sha256(OUT / 'protocol.json')
    records, previous, deps = [], [], []
    def forbidden(*args, **kwargs):
        raise AssertionError('Training operation forbidden during evaluation')
    with ExitStack() as stack:
        for obj, name in ((torch.optim.Optimizer, '__init__'), (torch.Tensor, 'backward'), (YOLO, 'train'), (YOLO, 'val')):
            stack.enter_context(patch.object(obj, name, forbidden))
        for seed, key in zip((7, 17, 27), KEYS):
            cp = OUT / 'training' / key / 'completion.json'
            c = verified(cp)
            e = verified(c['exposure_path'])
            if c['optimizer_steps'] != 480 or e['actual'] != p['schedules'][key]:
                raise ValueError('Training endpoint mismatch')
            oldpath = SOURCE / 'evaluation-v2' / f'visibility-452-{seed}.json'
            old = verified(oldpath)
            r = ev._evaluate(key, Path(c['weights']), paired, negatives, inputs)
            verified(ev.OUT / f'{key}.json')
            if len(r['rows']) != 48 or len(r['negative_rows']) != 48:
                raise ValueError('Incomplete evaluation')
            for field in ('rows', 'negative_rows'):
                if [(x['view_id'], x['variant'], x['image_sha256']) for x in r[field]] != [(x['view_id'], x['variant'], x['image_sha256']) for x in old[field]]:
                    raise ValueError('Comparison membership mismatch')
            records.append(r); previous.append(old)
            deps += [cp, oldpath, ev.OUT / f'{key}.json']
            print('COMPLETE', key, r['negative_summary'], flush=True)
    new, old = ev._aggregate(records), ev._aggregate(previous)
    changes = {v: {m: new[v][m]['mean'] - old[v][m]['mean'] for m in ('planned_instance_hit_rate', 'instance_recall', 'matched_precision')} for v in ev.VARIANTS}
    dest = ev.OUT / 'comparison.json'
    if dest.exists():
        return verified(dest)
    return write_record(dest, dict(status='numerical_complete_visual_review_pending', current=new, previous=old, changes=changes,
        matching_conflicts=sum(r['matching_conflicts'] for r in records), selected_candidate=None,
        training_admitted=False, promotable=False,
        limits=['Additional 120 negative exposures and 20 steps; not a fixed-budget comparison.',
                'Prior structure categories were inferred from frame subjects, not inspected prediction crops; no box-level visual approval is established.'],
        inputs={str(x): file_sha256(x) for x in deps + [Path(__file__).resolve()]}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--evaluate', action='store_true'); args = ap.parse_args()
    if args.evaluate:
        r = run(); print(r['status'], r['changes'], flush=True)
    else:
        print('NO_INFERENCE_NO_TRAINING')
