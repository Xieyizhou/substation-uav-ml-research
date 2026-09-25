"""Fail-closed, optimizer-free six-loader preflight and readiness receipt."""
import argparse
import fcntl
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.exposure_order_retention import *
from scripts.vision.freeze_order_retention import freeze
from scripts.vision.order_retention_runtime import preflight_cell,check_actual
from scripts.vision.review_exposure_order_retention import validate_decisions

MODULES=['tests.test_order_retention','tests.test_retention450_adapter','tests.test_retention_real_quota',
    'tests.test_retention_regression_review','tests.test_canonical_shutdown']

def validate_ready():
    path=OUT/'ready.json';verify_tree(path);r=read(path)
    if r['status']!='ready_for_training_not_started' or set(r['cells'])!=set(KEYS):raise ValueError('No complete readiness receipt')
    return read(OUT/'protocol.json')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'preflight.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'ready.json').exists():validate_ready();print('READY_REVALIDATED_NO_TRAINING');return
        p=freeze();verify_tree(OUT/'protocol.json')
        validate_decisions(read(OUT/'evidence.json'),read(OUT/'review.json')['decisions'])
        result=subprocess.run([sys.executable,'-m','unittest',*MODULES],cwd=ROOT,capture_output=True,text=True,check=True)
        print(result.stderr,flush=True)
        paths=[OUT/'protocol.json',OUT/'review.json',Path(__file__)]
        for key in KEYS:
            root=OUT/'preflight'/key;root.mkdir(parents=True,exist_ok=True);done=root/'completion.json'
            if done.exists():
                verify_tree(done);unit=read(done)
                if unit['protocol_identity']!=p['identity']:raise ValueError('Stale preflight protocol')
                check_actual(p,key,unit['actual'])
            else:
                attempts=list(root.glob('attempt-*'))
                if len(attempts)>=3:raise ValueError('Preflight attempt budget exhausted')
                attempt=root/f'attempt-{len(attempts)+1:03}';attempt.mkdir()
                try:
                    unit=preflight_cell(p,key)
                    frozen(done,dict(**unit,status='actual_loader_verified_without_optimization',protocol_identity=p['identity'],
                        inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),
                                str(ROOT/'scripts/vision/order_retention_runtime.py'):file_sha256(ROOT/'scripts/vision/order_retention_runtime.py')}))
                except BaseException:
                    save(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),workers=0,child_processes_started=0));raise
            paths.append(done);print('LOADER_PREFLIGHT_PASS',key,flush=True)
        tests=[ROOT/(name.replace('.','/')+'.py') for name in MODULES];paths+=tests
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
        if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
        frozen(OUT/'ready.json',dict(status='ready_for_training_not_started',cells=list(KEYS),
            training_started=False,optimizer_steps_executed=0,regression_output=result.stderr,whole_repository_tested=False,
            baseline=baseline,inputs={str(path):file_sha256(path) for path in paths}))
        print('READY_FOR_TRAINING_NOT_STARTED',flush=True)

if __name__=='__main__':main()
