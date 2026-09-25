"""Final loader-only acceptance, no training launch/readiness claim."""
from collections import Counter
from pathlib import Path
import subprocess,sys
from scripts.vision.preflight_frozen_multiscale import OUT,prior,validate
from scripts.vision.verify_experiment_baseline import verify as baseline_verify

def paired_checks(r):
    fixed,multi=(r['arms'][k]['batches'] for k in ('fixed','multiscale'))
    for a,b in zip(fixed,multi,strict=True):
        if a['members']!=b['members'] or a['source640_sha256']!=b['source640_sha256'] or a['full_supervision']!=b['full_supervision']:raise ValueError('Treatment changed members or supervision')
        for x in (a,b):
            if x['effective_shape']!=[6,3,x['size'],x['size']]:raise ValueError('Wrong effective shape')
        if b['size']==640 and a['effective_tensor_sha256']!=b['effective_tensor_sha256']:raise ValueError('640 control pixels differ')

def main():
    pp=OUT/'protocol.json';p=prior.read(pp);prior.verify(p);source=prior.read(p['source_protocol']);prior.verify(source)
    lp=OUT/'loader-completion.json';prior.verify(prior.read(lp));paths=[pp,lp,Path(__file__).resolve()];cells=[]
    for seed in (7,17,27):
        found=list((OUT/'checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
        if len(found)!=1:raise ValueError('Ambiguous or missing completion')
        r=prior.read(found[0]);validate(r,p,source);paired_checks(r);paths+=found
        if any(r.get(k) is not False for k in ('optimizer_created','backward_executed','training_validation_run')):raise ValueError('Forbidden training operation')
        lookup={m['member_id']:m for m in source['pool_rows']}
        for arm,c in r['arms'].items():
            classes=Counter();lineages=Counter();subsets=Counter()
            for mid in c['actual']:
                row=lookup[mid];classes.update(row['class_instances']);lineages[row['lineage_id']]+=1;subsets[row['subset']]+=1
            cells.append(dict(seed=seed,arm=arm,draws=len(c['actual']),batches=len(c['batches']),class_instances=dict(classes),lineage_exposure=dict(lineages),subset_exposure=dict(subsets),size_batches=dict(Counter(b['size'] for b in c['batches']))))
    suites=['tests.test_frozen_multiscale_runtime','tests.test_material_transfer_controls','tests.test_brightness_transfer','tests.test_order_retention','tests.test_unified_lighting_design']
    tests=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True,timeout=60)
    if tests.returncode:raise ValueError(tests.stdout+tests.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'completion.json',dict(status='loader_preflight_passed_training_not_started',cells=cells,
        total_actual_loads=16200,total_actual_batches=2700,historical_fixed_tensors_equal=True,
        training_started=False,training_authorized=False,training_ready=False,
        next_required='Integrate the same verified preprocess into an explicit training entry that consumes these receipts and validates each effective tensor. User authorization required for training.',
        interpolation_limit='960 is interpolated from brightness-processed 640, not direct-native 960 detail.',
        incidental_cache='Ultralytics rebuilt export/labels.cache; no image, label, protocol or weight modified. Cache is not scientific evidence.',
        tests_output=tests.stdout+tests.stderr,baseline=b,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(tests.stdout+tests.stderr);print('LOADERS_PASSED; TRAINING_NOT_STARTED')

if __name__=='__main__':main()
