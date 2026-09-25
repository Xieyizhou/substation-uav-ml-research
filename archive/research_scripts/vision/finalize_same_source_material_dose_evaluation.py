"""Verify explicit evidence and report; no training or automatic review decisions."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.evaluate_same_source_material_dose import OUT, KEYS, prior, complete, validate_record, verify_binding, baseline_verify
from scripts.vision.record_same_source_material_dose_review import validate


TESTS = ('test_same_source_material_dose_evaluation', 'test_same_source_material_dose_review',
         'test_material_late_fit', 'test_same_source_material_dose',
         'test_exposure_diagnosis', 'test_paired_visual_factors',
         'test_paired_visual_review', 'test_material_transfer_scope')


def main():
    dest = OUT/'evaluation/completion.json'
    if dest.exists():
        r=prior.read(dest); prior.verify(r); return r
    paths=[OUT/'evaluation/summary.json']
    summary=prior.read(paths[0]); prior.verify(summary)
    for key in KEYS:
        complete(key)
        p=OUT/'evaluation'/f'{key}.json'
        r=prior.read(p); validate_record(r,key); verify_binding(r,key)
        paths.extend([p,OUT/'training'/key/'completion.json'])
    evidence=OUT/'evaluation/error-review-v1/evidence.json'
    review=evidence.with_name('review.json')
    rs=evidence.with_name('review-summary.json')
    e,r,s=map(prior.read,(evidence,review,rs))
    for x in (e,r,s):prior.verify(x)
    validate(e,r['decisions'])
    if len(e['transitions'])!=720 or s['fp_predictions']!=3 or s['unique_loss_targets']!=29:
        raise ValueError('Incomplete error inventory')
    if summary['matching_conflicts']!=0:raise ValueError('Unresolved matching conflict')
    from scripts.vision.diagnose_same_source_material_dose_fit import OUT as FIT, freeze, validate as fit_validate
    fp=freeze()
    for key in KEYS:
        path=FIT/(key+'.json');fit_validate(prior.read(path),key,fp);paths.append(path)
    for name in ('protocol.json','summary.json'):
        path=FIT/name;prior.verify(prior.read(path));paths.append(path)
    from PIL import Image
    from scripts.vision.same_source_material_quality_v2 import pilot,expansion
    protocol=prior.read(OUT/'protocol.json'); rows={x['member_id']:x for x in protocol['pool_rows']}
    for module in (pilot,expansion):
        ep=module.DEST/'evidence.json';ev=prior.read(ep);prior.verify(ev);paths.append(ep)
        for event in ev['events']:
            if event['condition']!='gray_target_body':continue
            row=rows[event['source_id']+'-gray_target_body']
            with Image.open(row['image_path']) as a,Image.open(event['image_path']) as b:
                if a.size!=b.size or a.convert('RGB').tobytes()!=b.convert('RGB').tobytes():raise ValueError('Gray review/export pixels differ')
    paths.extend([OUT/'quality-v2.json',OUT/'entry-ready.json',OUT/'training/completion.json'])
    for path in paths:prior.verify(prior.read(path))
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:
        raise ValueError('Baseline integrity failure')
    test=subprocess.run([sys.executable,'-m','unittest',*('tests.'+x for x in TESTS)],
                        cwd=prior.ROOT,capture_output=True,text=True,timeout=120)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    report=prior.ROOT/'docs/results/ml_same_source_material_dose_20260913.md'
    paths.extend([evidence,review,rs,report,Path(__file__).resolve()])
    paths.extend(prior.ROOT/'tests'/f'{x}.py' for x in TESTS)
    return prior.frozen(dest,dict(status='development_evaluation_and_explicit_review_complete_not_selected',
        numerical_passed=summary['policy_results']['passed'],selected_candidate=None,
        matching_conflicts=0,baseline=baseline,review_decisions=len(r['decisions']),
        pending_ids=r['pending_ids'],tests=dict(command=TESTS,exit_code=test.returncode,
        stdout=test.stdout,stderr=test.stderr,scope='targeted_not_full_repository'),
        next_direction='source_isolated_material_data_design_with_original_retention_no_new_training',
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':print(main()['status'])
