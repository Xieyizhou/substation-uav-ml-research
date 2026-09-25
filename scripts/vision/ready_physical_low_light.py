"""Explicit data and actual-loader gates before any formal optimization."""
import subprocess,sys
from pathlib import Path
from scripts.vision.physical_low_light_design import OUT,KEYS,freeze,prior
from scripts.vision.preflight_physical_low_light import receipt
from scripts.vision.benchmark_closed_training_cpu_v2 import OUT as BENCH
from scripts.vision.exposure_order_retention import baseline_verify

TESTS=['tests.test_physical_low_light','tests.test_physical_low_light_design','tests.test_closed_budget_training','tests.test_closed_cpu_evaluation','tests.test_same_source_material_dose_evaluation']

def main():
    p=freeze();dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    deps=[OUT/'design.json',OUT/'quality-review.json',OUT/'export/manifest.json',OUT/'loader-completion.json',OUT/'entry-probes/completion.json',BENCH/'completion.json',Path(__file__).resolve()]
    b=prior.read(BENCH/'completion.json');prior.verify(b)
    if b['selected']!='dual4' or not next(x for x in b['results'] if x['config']=='dual4')['repeat_exact_state']:raise ValueError('CPU policy failure')
    for key in KEYS:
        r,rp=receipt(key,p);deps.append(rp)
        if len(r['actual'])!=6000 or len(r['batch_records'])!=1000 or any(r[k] for k in ('optimizer_created','backward_executed','validation_run')):raise ValueError('Actual loader gate failure')
    for key in ('R1000-7','L1000-7'):
        path=OUT/'entry-probes'/key/'completion.json';r=prior.read(path);prior.verify(r);deps.append(path)
        if r['status']!='actual_entry_prefix_probe_verified' or r['result']['optimizer_steps']!=10 or r['result']['effective_threads']!=4:raise ValueError('Entry probe failure')
    for d in deps:
        if d.suffix=='.json':prior.verify(prior.read(d))
    for name in ('train_physical_low_light.py','preflight_physical_low_light.py','probe_physical_low_light.py','physical_low_light_dataset.py','physical_low_light_design.py','closed_budget_engine_v2.py','closed_budget_runtime.py'):deps.append(Path(__file__).with_name(name))
    test=subprocess.run([sys.executable,'-m','unittest',*TESTS],cwd=prior.ROOT,capture_output=True,text=True)
    if test.returncode:raise ValueError('Targeted regression failure '+test.stdout+test.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Fixed baseline integrity failure')
    deps.extend(prior.ROOT/(t.replace('.','/')+'.py') for t in TESTS)
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),cpu_selection='dual4',threads=4,parallel=2,baseline=baseline,
        tests=dict(modules=TESTS,stdout=test.stdout,stderr=test.stderr,returncode=test.returncode,whole_repository_claim=False),inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print(main()['status'])
