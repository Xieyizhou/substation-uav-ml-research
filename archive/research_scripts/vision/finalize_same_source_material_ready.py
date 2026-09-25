"""Gate explicit existing reviews, exact schedule and actual loaders before training."""
import subprocess,sys
from pathlib import Path
from collections import Counter
from scripts.vision.train_same_source_material_dose import OUT,KEYS,contract,prior,reference
from scripts.vision.same_source_material_quality_v2 import main as quality
from scripts.vision.freeze_same_source_material_dose import check
from scripts.vision.exposure_order_retention import baseline_verify

def main():
    q=quality()
    if q['gaps'] or len(q['members'])!=48:raise ValueError('Unresolved quality evidence')
    paths=[OUT/'quality-v2.json',OUT/'protocol.json',Path(__file__).resolve()];stats={}
    for key in KEYS:
        p,_,actual=contract(key);seed=key.split('-')[-1];old,_,expected=reference.contract('T-'+seed)
        check(p,old);rows={r['member_id']:r for r in p['pool_rows']}
        if sum(m['label_count'] for m in q['members'])!=136:raise ValueError('Incomplete review supervision inventory')
        for i,(before,after) in enumerate(zip(expected['batch_records'],actual['batch_records'],strict=True)):
            if before['full_supervision']!=after['full_supervision']:raise ValueError('Actual full supervision differs')
            if not any(pos//6==i for pos in p['changed_positions'][key]) and before['image_tensor_sha256']!=after['image_tensor_sha256']:raise ValueError('Untreated tensor changed')
        before=old['schedules']['T-'+seed];after=p['schedules'][key]
        for field in ('lineage_id','subset'):
            if Counter(rows[m][field] for m in before)!=Counter(rows[m][field] for m in after):raise ValueError('Source/subset budget changed')
        omitted=sorted(set(before)-set(after))
        stats[key]=dict(original_members_zero_exposure=omitted,changed_positions=p['changed_positions'][key],actual_draws=len(actual['actual']),
            actual_pixel_changed_positions=[i for i,(a,b) in enumerate(zip(expected['brightness_log'],actual['brightness_log'],strict=True)) if a['after']!=b['after']])
        paths.extend((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    tests=('test_same_source_material_dose','test_material_dose_feasibility','test_material_fixed_sequence_lr_review','test_exposure_diagnosis')
    run=subprocess.run([sys.executable,'-m','unittest',*('tests.'+x for x in tests)],cwd=prior.ROOT,capture_output=True,text=True,timeout=120)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    paths.extend(prior.ROOT/'tests'/f'{x}.py' for x in tests)
    paths.extend((prior.ROOT/'scripts/vision/train_same_source_material_dose.py',prior.ROOT/'docs/plans/ml_same_source_material_dose_20260913.md'))
    dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),stats=stats,baseline=baseline,
        review_policy='Reuse 136 identity-valid explicit source/variant label observations, no new approval generated.',
        limits=['Partial content reviews retained; no pixel visibility certification or universal quality claim.',
                'Original appearance removal explicitly authorized; same source supervision retained.'],
        tests=dict(exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr),inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':print(main()['status'])
