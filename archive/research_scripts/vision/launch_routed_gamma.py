"""Additional contract regression gate, then explicit three-seed automatic run."""
import argparse
from pathlib import Path
import subprocess
import sys
from scripts.vision import routed_gamma_control as a
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run(train=False):
    a.configure();a.runtime.launch_gate(create=True)
    t=subprocess.run([sys.executable,'-m','unittest','tests.test_routed_gamma_contract','-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise RuntimeError(t.stdout+t.stderr)
    dest=a.OUT/'gamma-contract-readiness.json'
    paths=[a.OUT/'protocol.json',a.OUT/'launch-readiness.json',Path(__file__),Path('tests/test_routed_gamma_contract.py')]
    if dest.exists():a.checked(dest)
    else:write_record(dest,dict(status='contract_tests_passed',test_output=t.stdout+t.stderr,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))
    if train:a.runtime.train()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');args=p.parse_args();run(args.train)
