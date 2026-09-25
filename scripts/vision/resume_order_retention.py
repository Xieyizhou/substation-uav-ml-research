"""Compatibility repair: immutable protocol mirror for the existing evaluator."""
import fcntl
import shutil
import subprocess
import sys
from pathlib import Path
from scripts.vision.exposure_order_retention import OUT,ROOT,KEYS,read,save,file_sha256,verify_tree
from scripts.vision.preflight_order_retention import validate_ready
from scripts.vision.train_order_retention import isolated

def bind_protocol(source,target):
    if target.exists():
        if target.read_bytes()!=source.read_bytes():raise ValueError('Existing evaluation protocol differs; do not overwrite')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
    if file_sha256(source)!=file_sha256(target):raise ValueError('Protocol mirror mismatch')

def main():
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_ready()
        result=subprocess.run([sys.executable,'-m','unittest','tests.test_order_protocol_repair'],cwd=ROOT,capture_output=True,text=True,check=True)
        source=OUT/'protocol.json';mirror=OUT/'training/protocol.json'
        bind_protocol(source,mirror)
        receipt=OUT/'evaluation-path-repair-v1.json'
        paths=[source,mirror,Path(__file__),ROOT/'tests/test_order_protocol_repair.py']
        if receipt.exists():verify_tree(receipt)
        else:save(receipt,dict(status='evaluation_protocol_mirror_verified',canonical_protocol=str(source),mirror=str(mirror),
            unchanged_training_and_evaluation_behavior=True,regression_output=result.stderr,
            inputs={str(p):file_sha256(p) for p in paths}))
        print('PROTOCOL_PATH_REPAIRED_EXISTING_WEIGHTS_REUSED',flush=True)
        for key in KEYS:
            bind_protocol(source,mirror)
            done=OUT/'training'/key/'evaluated.json'
            if done.exists():verify_tree(done)
            else:isolated(key)
            save(OUT/'resume-progress.json',dict(status='in_progress' if key!=KEYS[-1] else 'all_six_evaluated_pending_review',
                last_completed_cell=key,inputs={str(receipt):file_sha256(receipt),str(done):file_sha256(done)}))
            print('EVALUATED',key,flush=True)

if __name__=='__main__':main()
