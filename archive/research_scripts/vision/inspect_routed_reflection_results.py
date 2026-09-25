"""Freeze this reflection arm's errors without generating review decisions."""
from pathlib import Path
from types import SimpleNamespace
from scripts.vision import routed_reflection_control as arm
from scripts.vision import mild_routed_contrast_control as prior
from scripts.vision import inspect_mild_routed_results as builder
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=arm.OUT/'audit-v1'

def run():
    builder.arm=SimpleNamespace(OUT=arm.OUT,KEYS=arm.KEYS,SOURCE=arm.SOURCE,SOURCE_KEYS=arm.SOURCE_KEYS,
        GLOBAL=prior.GLOBAL,GLOBAL_KEYS=prior.GLOBAL_KEYS,checked=arm.checked)
    builder.OUT=OUT;builder.run()
    paths=[OUT/'evidence.json',OUT/'material-evidence.json',arm.OUT/'evaluation-v1/error-review-v1/evidence.json',Path(__file__),Path(builder.__file__)]
    for p in paths[:3]:arm.checked(p)
    dest=OUT/'evidence-build-receipt.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='current_reflection_errors_frozen_visual_review_pending',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
