"""Recompute stored matching and bind report, tests, and inference dependencies."""
import subprocess
import sys
from scripts.vision.evaluate_whole_image_hold import *

def audit():
    p=ready();rp=Path(p['evaluation']['paired_review']);np=Path(p['evaluation']['negative_review'])
    paired,_=paired_truth(read(rp)['frames']);neg=read(np)['frames']
    pairs={(r['view_id'],r['variant']):(r,t) for r,t in paired}
    negatives={(r['view_id'],r['variant']):r for r in neg}
    paths=[rp,np,OUT/'evaluation/summary.json',Path(__file__)]
    for key in KEYS:
        ep=OUT/'evaluation'/f'{key}.json';r=read(ep);verify(r);paths.append(ep)
        if len(r['rows'])!=48 or {(x['view_id'],x['variant']) for x in r['rows']}!=set(pairs):raise ValueError('Paired membership mismatch')
        for x in r['rows']:
            row,t=pairs[x['view_id'],x['variant']]
            if x!=score(row,t,x['predictions'],x['low_predictions']):raise ValueError('Stored scoring differs')
        if len(r['negative_rows'])!=48 or {(x['view_id'],x['variant']) for x in r['negative_rows']}!=set(negatives):raise ValueError('Negative membership mismatch')
        for x in r['negative_rows']:
            if x['image_sha256']!=negatives[x['view_id'],x['variant']]['image_sha256'] or x['frame_has_prediction']!=bool(x['predictions']):raise ValueError('Negative identity/count mismatch')
        if r['summary']!={v:summary([x for x in r['rows'] if x['variant']==v]) for v in VARIANTS}:raise ValueError('Summary mismatch')
    tests=['tests.test_whole_image_hold_train','tests.test_whole_image_hold','tests.test_matched_appearance_training','tests.test_structure_fit','tests.test_structure_fit_integrity']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    paths += [ROOT/(m.replace('.','/')+'.py') for m in tests]
    paths += [ROOT/'scripts/vision'/name for name in ['evaluate_exposure_diagnosis.py','exposure_metrics.py','evaluate_paired_visual_factors.py','evaluate_visual_augmentation_abcd.py','analyze_recovery_paired_calibration.py']]
    paths.append(ROOT/'docs/results/ml_whole_image_hold_evaluation_20260909.md')
    frozen(OUT/'evaluation/completion.json',dict(status='numerical_diagnosis_complete_no_candidate',
        new_visual_reviews_generated=False,selected_candidate=None,regression_output=run.stderr,whole_repository_tested=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('AUDIT_PASSED_NO_CANDIDATE')

if __name__=='__main__':audit()
