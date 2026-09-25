"""Sign readiness only after actual traversal, deterministic CPU probe and tests."""
from pathlib import Path
import subprocess,sys
from scripts.vision.closed_budget_design import OUT,KEYS,freeze,prior
from scripts.vision.preflight_closed_budget import receipt
from scripts.vision.benchmark_closed_training_cpu_v2 import OUT as BENCH,CONFIGS
from scripts.vision.exposure_order_retention import baseline_verify

TESTS=['tests.test_closed_budget_design','tests.test_closed_budget_training','tests.test_closed_pool_fit','tests.test_closed_cpu_evaluation','tests.test_closed_source_error_review','tests.test_same_source_material_dose_evaluation']

def main():
    p=freeze();dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    benchmark=prior.read(BENCH/'completion.json');prior.verify(benchmark)
    selected=next(r for r in benchmark['results'] if r['config']==benchmark['selected'])
    if not selected['repeat_exact_state']:raise ValueError('Unrepeatable CPU policy')
    deps=[OUT/'design.json',OUT/'loader-completion.json',BENCH/'completion.json',Path(__file__).resolve()]
    prior.verify(prior.read(OUT/'loader-completion.json'))
    for key in KEYS:
        r,rp=receipt(key,p);deps.append(rp)
        if any(r[k] for k in ('optimizer_created','backward_executed','validation_run')):raise ValueError('Preflight ran training')
    for name in ('closed_budget_engine_v2.py','closed_budget_runtime.py','train_closed_budget.py','preflight_closed_budget.py'):deps.append(Path(__file__).with_name(name))
    test=subprocess.run([sys.executable,'-m','unittest',*TESTS],cwd=prior.ROOT,capture_output=True,text=True)
    if test.returncode:raise ValueError('Regression failed: '+test.stdout+test.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity')
    deps.extend(prior.ROOT/(t.replace('.','/')+'.py') for t in TESTS)
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),cpu_selection=benchmark['selected'],
        threads=CONFIGS[benchmark['selected']][0],parallel=CONFIGS[benchmark['selected']][1],baseline=baseline,
        tests=dict(modules=TESTS,stdout=test.stdout,stderr=test.stderr,returncode=test.returncode,whole_repository_claim=False),
        inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print(main()['status'])
