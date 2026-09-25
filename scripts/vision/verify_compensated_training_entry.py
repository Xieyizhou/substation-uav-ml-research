"""Sign only the tested explicit entry, never start optimization."""
from pathlib import Path
import subprocess,sys
from scripts.vision.train_compensated_material import OUT,prior,KEYS,contract
from scripts.vision.evaluate_scale_endpoints import baseline_verify

def main():
    for key in KEYS:contract(key)
    suites=['tests.test_compensated_training_entry','tests.test_compensated_loader_receipts','tests.test_compensated_material_sequences']
    r=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if r.returncode:raise ValueError(r.stdout+r.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline')
    paths=[OUT/'protocol.json',OUT/'loader-completion.json',OUT.parent/'dataset-completion.json',Path(__file__),prior.ROOT/'scripts/vision/train_compensated_material.py',prior.ROOT/'scripts/vision/frozen_multiscale_runtime.py',prior.ROOT/'scripts/vision/brightness_transfer_runtime.py',prior.ROOT/'scripts/vision/order_retention_runtime.py']
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'entry-ready.json',dict(status='training_entry_verified_not_started',cells=list(KEYS),baseline=b,regression_output=r.stdout+r.stderr,
        training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('ENTRY6_VERIFIED; NO_TRAINING')

if __name__=='__main__':main()
