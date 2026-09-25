"""Audit benchmark identities and numerical observations, not model admission."""
from pathlib import Path
import subprocess
import sys
import yaml
from scripts.vision.benchmark_reviewed_hold_device import OUT,CELLS,prior
from scripts.vision.test_body_material_applicability import baseline_verify

def main():
    p=prior.read(OUT/'protocol.json');prior.verify(p)
    s=prior.read(OUT/'summary.json');prior.verify(s)
    rs={};paths=[OUT/'protocol.json',OUT/'summary.json',Path(__file__).resolve()]
    for c in CELLS:
        cp=next((OUT/c).glob('attempt-*/completion.json'));r=prior.read(cp);prior.verify(r);rs[c]=r
        args=yaml.safe_load((cp.parent/'run/args.yaml').read_text())
        expected={**p['config'],'device':c.split('-')[0]}
        if any(args[k]!=v for k,v in expected.items()):raise ValueError('Actual configuration drift')
        if r['draw_count']!=180 or len(r['steps'])!=30:raise ValueError('Incomplete cell')
        paths += [cp,cp.parent/'log.txt']
    def compare(a,b):
        x,y=rs[a],rs[b]
        return dict(state_hashes_identical=x['state_hashes']==y['state_hashes'],
            max_absolute_loss_delta=max(abs(u['loss'][k]-v['loss'][k]) for u,v in zip(x['steps'],y['steps']) for k in u['loss']),
            first_step_max_absolute_loss_delta=max(abs(x['steps'][0]['loss'][k]-y['steps'][0]['loss'][k]) for k in x['steps'][0]['loss']))
    tests=['tests.test_device_acceleration_benchmark','tests.test_brightness_lr_retention','tests.test_whole_image_hold_train','tests.test_reviewed_hold_finalization']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    paths += [prior.ROOT/(n.replace('.','/')+'.py') for n in tests]
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    report=prior.ROOT/'docs/results/ml_device_acceleration_benchmark_20260910.md';paths.append(report)
    prior.frozen(OUT/'completion.json',dict(status='engineering_benchmark_complete_formal_migration_not_approved',
        timings=s,repeat_cpu=compare('cpu-1','cpu-2'),repeat_mps=compare('mps-1','mps-2'),cross_device=compare('cpu-1','mps-1'),
        determinism_warnings=['scatter_reduce_mps','index_put_with_accumulate_mps'],
        interpretation='Short repeated equality is observed, not guaranteed; no quality/450-step equivalence established.',
        regression=dict(output=t.stderr,whole_repository_tested=False),baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('BENCHMARK_AUDITED',s['speedup_cpu_over_mps'])

if __name__=='__main__':main()
