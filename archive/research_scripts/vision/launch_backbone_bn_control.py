"""Explicit launch after complete loader, model, regression and identity checks."""
import argparse
import subprocess
import sys
from pathlib import Path
from scripts.vision import routed_backbone_bn_control as arm
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def gate():
    arm.configure();p,m=arm.prepare()
    if m['actual_draws_verified']!=8640 or m['optimizer_created'] or m['backward_executed']:raise ValueError('Invalid model preflight')
    ready=arm.runtime.launch_gate(create=True)
    if ready['integrity']['pinned_files_verified']!=40:raise ValueError('Fixed40 missing')
    tests=['tests.test_backbone_bn_update_policy','tests.test_routed_backbone_bn_control']
    result=subprocess.run([sys.executable,'-m','unittest',*tests,'-q'],capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    paths=[arm.OUT/n for n in ('protocol.json','entry-ready.json','model-preflight.json','launch-readiness.json')]+[Path(__file__),Path('tests/test_routed_backbone_bn_control.py')]
    path=arm.OUT/'pretraining-completion.json'
    if path.exists():return arm.checked(path)
    return write_record(path,dict(status='ready_for_training_not_started',actual_draws=8640,tests=result.stdout+result.stderr,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');a=ap.parse_args();print(gate()['status'],flush=True)
    if a.train:arm.runtime.train()
