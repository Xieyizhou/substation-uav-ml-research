"""Validate the explicit three-seed optimizer entry without starting it."""
from pathlib import Path
import subprocess,sys
from scripts.vision.train_gray_body_control import OUT,prior,KEYS,contract
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def main():
    for key in KEYS:contract(key)
    suites=['tests.test_gray_body_training_entry','tests.test_gray_body_control','tests.test_transfer_pilot_review',
            'tests.test_material_transfer_scope','tests.test_compensated_loader_receipts']
    r=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=120)
    if r.returncode:raise ValueError(r.stdout+r.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths=[OUT/'protocol.json',OUT/'route-ready.json',OUT/'loader-completion.json',Path(__file__).resolve()]
    paths += [prior.ROOT/'scripts/vision'/f for f in ('train_gray_body_control.py','frozen_multiscale_runtime.py',
        'brightness_transfer_runtime.py','order_retention_runtime.py','finalize_gray_body_route.py','freeze_gray_body_control.py')]
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    dest=OUT/'entry-ready.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='training_entry_verified_not_started',cells=list(KEYS),baseline=b,
        regression_output=r.stdout+r.stderr,training_started=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('THREE_CELL_ENTRY_VERIFIED_NO_TRAINING',r.stdout+r.stderr)


if __name__=='__main__':main()
