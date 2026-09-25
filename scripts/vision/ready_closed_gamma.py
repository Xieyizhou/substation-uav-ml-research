"""Readiness requires explicit review, actual traversal and optimization probes."""
from pathlib import Path
import subprocess,sys
from scripts.vision.closed_gamma_design import OUT,KEYS,SOURCE,freeze,prior
from scripts.vision.preflight_closed_gamma import receipt
from scripts.vision.review_closed_gamma import main as review
from scripts.vision.benchmark_closed_training_cpu_v2 import OUT as BENCH,CONFIGS
from scripts.vision.exposure_order_retention import baseline_verify
from scripts.vision.train_closed_budget import complete as reference_complete

TESTS=['tests.test_closed_gamma','tests.test_closed_gamma_readiness','tests.test_closed_budget_training','tests.test_closed_cpu_evaluation','tests.test_closed_source_error_review','tests.test_same_source_material_dose_evaluation']

def main():
    p=freeze();review();dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    benchmark=prior.read(BENCH/'completion.json');prior.verify(benchmark)
    selected=next(r for r in benchmark['results'] if r['config']==benchmark['selected'])
    if not selected['repeat_exact_state'] or benchmark['selected']!='dual4':raise ValueError('CPU policy invalid')
    deps=[OUT/'design.json',OUT/'loader-completion.json',OUT/'augmentation-review/review.json',BENCH/'completion.json',Path(__file__).resolve(),OUT/'entry-probes/completion.json']
    for mode in ('identity','gamma'):
        path=OUT/'entry-probes'/mode/'completion.json';r=prior.read(path);prior.verify(r);deps.append(path)
        if r['status']!='real_ten_step_probe_verified' or r['mode']!=mode or r['result']['optimizer_steps']!=10 or r['result']['effective_threads']!=4:raise ValueError('Invalid entry probe')
    for key in KEYS:
        r,rp=receipt(key,p);deps.append(rp)
        if any(r[k] for k in ('optimizer_created','backward_executed','validation_run')) or not r['identity_batches_exact'] or not r['full_supervision_exact']:raise ValueError('Invalid loader preflight')
        ref='B900-'+key.split('-')[-1];reference_complete(ref);deps.append(SOURCE/'training'/ref/'completion.json')
    for name in ('closed_gamma_engine.py','closed_gamma_runtime.py','train_closed_gamma.py','preflight_closed_gamma.py','review_closed_gamma.py','probe_closed_gamma.py'):deps.append(Path(__file__).with_name(name))
    for d in deps:
        if d.suffix=='.json':prior.verify(prior.read(d))
    test=subprocess.run([sys.executable,'-m','unittest',*TESTS],cwd=prior.ROOT,capture_output=True,text=True)
    if test.returncode:raise ValueError('Regression failed: '+test.stdout+test.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity')
    deps.extend(prior.ROOT/(t.replace('.','/')+'.py') for t in TESTS)
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),cpu_selection=benchmark['selected'],threads=4,parallel=2,baseline=baseline,
        tests=dict(modules=TESTS,stdout=test.stdout,stderr=test.stderr,returncode=test.returncode,whole_repository_claim=False),inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print(main()['status'])
