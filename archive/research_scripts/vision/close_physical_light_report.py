"""Read-only optimization-prefix audit and report binding; no inference/training."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.physical_low_light_capture import OUT, prior
from scripts.vision.exposure_order_retention import baseline_verify
from scripts.vision.ready_physical_low_light import TESTS
from scripts.vision.record_physical_light_error_review import validate

def main():
    report=prior.ROOT/'docs/results/ml_physical_low_light_evaluation_20260914.md'
    deps=[report,Path(__file__).resolve(),OUT/'evaluation-completion.json',OUT/'evaluation/summary.json',OUT/'evaluation/error-review-v1/review.json',OUT/'evaluation/error-review-v1/evidence.json']
    for p in deps[2:]: prior.verify(prior.read(p))
    validate(prior.read(deps[5]),prior.read(deps[4])['decisions'])
    prefix=[]
    for family in ('R1000','L1000'):
        for seed in (7,17,27):
            xs=[]
            for root,key in ((OUT,f'{family}-{seed}'),(OUT.parent,f'B900-{seed}')):
                cp=root/'training'/key/'completion.json';c=prior.read(cp);prior.verify(c)
                xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x);xs.append(x);deps.extend([cp,xp])
            a,b=xs
            if len(a['step_records'])!=1000 or len(b['step_records'])!=900:raise ValueError('Step count')
            if any(x['loss_items']!=y['loss_items'] for x,y in zip(a['step_records'][:900],b['step_records'],strict=True)):raise ValueError('Loss prefix differs')
            prefix.append(dict(cell=f'{family}-{seed}',steps=900,recorded_losses_exact=True))
    tests=TESTS+['tests.test_physical_light_error_review']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=prior.ROOT,capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    deps.extend(prior.ROOT/(n.replace('.','/')+'.py') for n in tests)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    return prior.frozen(OUT/'report-completion.json',dict(status='evaluation_review_and_report_complete_no_candidate',prefix_loss_audit=prefix,
        limitation='Recorded losses, samples and input tensors match; no intermediate checkpoint identity claim.',
        tests=dict(modules=tests,returncode=t.returncode,stdout=t.stdout,stderr=t.stderr,whole_repository_claim=False),baseline=baseline,
        selected_candidate=None,inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':print(main()['status'])
