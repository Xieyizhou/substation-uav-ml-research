"""Explicit existing-weight fitting diagnosis for the scale arm; never trains."""
import argparse
from pathlib import Path
from scripts.vision import diagnose_routed_amplitude_fit as engine
from scripts.vision import routed_scale_control as arm
from scripts.vision import clear_context_training as runtime
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT = arm.OUT.parent / 'routed-scale-fit-diagnosis-v1'
KEYS = tuple(arm.KEYS)

def freeze():
    engine.OUT = OUT
    engine.MODELS = {k: arm.OUT for k in KEYS}
    engine.KEYS = KEYS
    for key in KEYS:
        arm.verified_unit(key)
    engine.freeze()
    paths = [OUT/'protocol.json', Path(__file__), Path(engine.__file__),
             arm.OUT/'evaluation-v1/summary.json']
    prior = arm.OUT.parent/'routed-amplitude-fit-diagnosis-v1'
    for key in arm.SOURCE_KEYS:
        for suffix in ('.json', '-verified.json'):
            p = prior/(key+suffix)
            arm.checked(p)
            paths.append(p)
    dest = OUT/'research-entry.json'
    if dest.exists():
        return arm.checked(dest)
    return write_record(dest, dict(
        status='frozen_existing_weights_fit_research_no_training',
        question='Does scale-arm degradation also affect original exposed material members?',
        limits='Unaugmented training-member fitting only; not augmented-tensor fitting or generalization. No new data or training.',
        training_admitted=False, promotable=False,
        inputs={str(p.resolve()):file_sha256(p) for p in paths}))

def run():
    freeze()
    runtime.KEYS = KEYS
    runtime.MODULE = 'scripts.vision.diagnose_routed_scale_fit'
    runtime.parallel('--worker', OUT/'logs')
    for key in KEYS:
        arm.checked(OUT/(key+'-verified.json'))
    engine.base.OUT = OUT
    engine.base.KEYS = KEYS
    print(engine.base.summarize()['groups'], flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--worker', choices=KEYS)
    a = ap.parse_args()
    if a.worker:
        freeze()
        engine.worker(a.worker)
    elif a.infer:
        run()
    else:
        freeze()
        print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
