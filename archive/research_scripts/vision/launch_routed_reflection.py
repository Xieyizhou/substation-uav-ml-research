"""Supplemental immutable execution identity and explicit launch, no default train."""
import argparse
import subprocess
import sys
from pathlib import Path
from scripts.vision import routed_reflection_control as arm
from scripts.vision import mild_routed_contrast_control as report
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def prepare():
    arm.configure();arm.freeze();arm.runtime.preflight()
    path=arm.OUT/'execution-identity.json'
    if path.exists():return arm.checked(path)
    command=[sys.executable,'-m','unittest','tests.test_routed_reflection','tests.test_routed_reflection_review_gate','-q']
    r=subprocess.run(command,text=True,capture_output=True,timeout=180)
    if r.returncode:raise ValueError(r.stdout+r.stderr)
    files=[Path(__file__),Path(arm.__file__),Path(report.__file__),Path('tests/test_routed_reflection_review_gate.py'),Path('tests/test_routed_reflection.py'),arm.OUT/'protocol.json',arm.OUT/'entry-ready.json',arm.gray.OUT/'audit-v1/report-receipt.json']
    return write_record(path,dict(status='execution_and_extra_review_gate_tests_verified',test_output=r.stdout+r.stderr,
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in files}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');a=ap.parse_args();prepare()
    if a.train:
        arm.runtime.train();arm.checked(arm.OUT/'execution-identity.json')
    else:print('SUPPLEMENTAL_PREFLIGHT_ONLY_NO_TRAINING')
