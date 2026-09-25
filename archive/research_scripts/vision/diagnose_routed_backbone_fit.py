"""Next research step: exposed-material fitting with existing frozen-backbone weights."""
import argparse
from pathlib import Path
from scripts.vision import diagnose_routed_amplitude_fit as engine
from scripts.vision import routed_backbone_control as arm
from scripts.vision import clear_context_training as runtime
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=arm.OUT.parent/'routed-backbone-fit-diagnosis-v1'
KEYS=tuple(arm.KEYS)

def configure():
    engine.OUT=OUT;engine.MODELS={k:arm.OUT for k in KEYS};engine.KEYS=KEYS

def freeze():
    configure()
    for key in KEYS:arm.verified_unit(key)
    engine.freeze()
    # Reuse valid routed fitting predictions as the direct comparison, not re-infer them.
    prior=arm.OUT.parent/'routed-amplitude-fit-diagnosis-v1'
    paths=[OUT/'protocol.json',Path(__file__),Path(engine.__file__),arm.OUT/'evaluation-v1/summary.json']
    for key in arm.SOURCE_KEYS:
        for q in (prior/f'{key}.json',prior/f'{key}-verified.json'):
            arm.checked(q);paths.append(q)
    for key in KEYS:paths.append(arm.OUT/'training'/key/'backbone-verification.json')
    dest=OUT/'research-entry.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='frozen_existing_weights_fit_research_no_training',
        question='Does frozen-backbone degradation also appear on the identical actually exposed material training members, or mainly on development conditions?',
        limits='All 144 material members and all labels; three seeds, no score-based selection. Fitting is not generalization. No new training, collection, label edits or sealed testing.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

def run():
    freeze();runtime.KEYS=KEYS;runtime.MODULE='scripts.vision.diagnose_routed_backbone_fit'
    runtime.parallel('--worker',OUT/'logs')
    for key in KEYS:arm.checked(OUT/f'{key}-verified.json')
    engine.base.OUT=OUT;engine.base.KEYS=KEYS
    result=engine.base.summarize();print(result['status'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:freeze();engine.worker(a.worker)
    elif a.infer:run()
    else:freeze();print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
