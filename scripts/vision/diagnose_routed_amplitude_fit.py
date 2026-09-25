"""Existing-weight, full-label material fitting comparison; never train."""
import argparse
from pathlib import Path
from scripts.vision import diagnose_lr_material_fit as base
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import mild_routed_contrast_control as mild
from scripts.vision.locked_cpu_threads import locked_threads
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT = routed.OUT.parent / 'routed-amplitude-fit-diagnosis-v1'
MODELS = {**{k: routed.OUT for k in routed.KEYS},
          **{k: mild.OUT for k in mild.KEYS}}
KEYS = tuple(MODELS)


def freeze():
    OUT.mkdir(exist_ok=True)
    dest = OUT / 'protocol.json'
    if dest.exists():
        return base.checked(dest)
    paths = [Path(__file__), Path(base.__file__),
             Path('scripts/vision/locked_cpu_threads.py'),
             Path('scripts/vision/diagnose_reviewed_training_fit.py'),
             Path('scripts/vision/evaluate_reactor_visibility_expansion.py'),
             Path('scripts/vision/evaluate_paired_visual_factors.py'),
             Path('scripts/vision/exposure_metrics.py')]
    for key, root in MODELS.items():
        p = base.checked(root / 'protocol.json')
        c = base.checked(root / 'training' / key / 'completion.json')
        e = base.checked(Path(c['exposure_path']))
        if e['actual'] != p['schedules'][key]:
            raise ValueError('Actual exposure mismatch: ' + key)
        paths += [root / 'protocol.json', root / 'training' / key / 'completion.json',
                  Path(c['exposure_path']), Path(c['weights'])]
        for row in p['pool_rows']:
            if row['variant'] in base.VARIANTS:
                for kind in ('image', 'label'):
                    path = Path(row[kind + '_path'])
                    if file_sha256(path) != row[kind + '_sha256']:
                        raise ValueError('Stale material member')
                    paths.append(path)
    return write_record(dest, dict(
        status='frozen_existing_weight_fit_only', models=list(KEYS),
        variants=list(base.VARIANTS), cpu_threads=4, workers=2,
        confidence=[.37, .001], input_size=640, nms_iou=.7, max_det=300,
        matching='same_class_one_to_one_iou_at_least_0.5',
        training_admitted=False, promotable=False,
        inputs={str(p.resolve()): file_sha256(p) for p in paths}))


def worker(key):
    freeze()
    base.TRAIN = MODELS[key]
    base.OUT = OUT
    with locked_threads(4) as events:
        result = base.worker(key)
    dest = OUT / (key + '-verified.json')
    if dest.exists():
        return base.checked(dest)
    return write_record(dest, dict(
        status='existing_weight_inference_complete', key=key,
        rows=len(result['rows']), thread_events=events,
        training_admitted=False, promotable=False,
        inputs={str(p.resolve()): file_sha256(p)
                for p in (OUT / 'protocol.json', OUT / (key + '.json'))}))


def run():
    freeze()
    from scripts.vision import clear_context_training as runtime
    runtime.KEYS = KEYS
    runtime.MODULE = 'scripts.vision.diagnose_routed_amplitude_fit'
    runtime.parallel('--worker', OUT / 'logs')
    for key in KEYS:
        base.checked(OUT / (key + '-verified.json'))
    base.OUT = OUT
    base.KEYS = KEYS
    print(base.summarize()['groups'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--worker', choices=KEYS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.infer:
        run()
    else:
        freeze()
        print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
