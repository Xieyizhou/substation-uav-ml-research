"""Recompute scores and require explicit reviews before publishing completion."""
import subprocess,sys
from pathlib import Path
from scripts.vision.lineage_capped_control import OUT,RUN,ROOT,KEYS,ready,read,verify,frozen,file_sha256
from scripts.vision.train_lineage_capped_v2 import completed
from scripts.vision.run_fixed_budget_diagnosis import paired_truth,score,summary,VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.prepare_whole_image_hold import baseline_verify

def verify_decisions(e,ds):
    expected={}
    for r in e['events']:
        for i in range(len(r['predictions']) if r['kind']=='FP' else 1):expected[f"{r['event_id']}:{i}"]=(r,i)
    if len(ds)!=len(expected) or {d['review_id'] for d in ds}!=set(expected):raise ValueError('Missing/duplicate review')
    for d in ds:
        r,i=expected[d['review_id']]
        if d['image_sha256']!=r['source']['image_sha256'] or d['evidence_sha256']!=r['evidence_sha256'] or d['crop_sha256']!=r['crops'][i]['sha256']:raise ValueError('Stale review evidence')
        if not d['reason'] or d['review_nature']!='AI辅助审核' or not d['reviewed_at']:raise ValueError('Incomplete observation')
        if d['pixel_visibility_certified'] is not False:raise ValueError('No instance-mask certification')
        if r['kind']=='FP' and d['prediction']!=r['predictions'][i]:raise ValueError('Prediction changed')
        if r['kind']=='LOSS' and d['truth']!=r['truth']:raise ValueError('Truth changed')

def main():
    p=ready();rp=Path(p['evaluation']['paired_review']);np=Path(p['evaluation']['negative_review']);paired,_=paired_truth(read(rp)['frames'])
    pairs={(r['view_id'],r['variant']):(r,t) for r,t in paired};neg={(r['view_id'],r['variant']):r for r in read(np)['frames']}
    records={};paths=[OUT/'protocol.json',OUT/'ready.json',rp,np,Path(__file__)]
    for key in KEYS:
        completed(key,p);path=OUT/'evaluation'/f'{key}.json';r=read(path);verify(r);records[key]=r;paths.extend([path,OUT/'training'/key/'adapter-binding.json'])
        if r['matching_conflicts'] or len(r['rows'])!=48 or len(r['negative_rows'])!=48:raise ValueError('Incomplete/conflicted evaluation')
        if {(x['view_id'],x['variant']) for x in r['rows']}!=set(pairs) or {(x['view_id'],x['variant']) for x in r['negative_rows']}!=set(neg):raise ValueError('Member mismatch')
        for x in r['rows']:
            source,t=pairs[x['view_id'],x['variant']]
            if x!=score(source,t,x['predictions'],x['low_predictions']):raise ValueError('Matching/diagnosis changed')
        for x in r['negative_rows']:
            if x['image_sha256']!=neg[x['view_id'],x['variant']]['image_sha256'] or x['frame_has_prediction']!=bool(x['predictions']):raise ValueError('Negative count drift')
        if r['summary']!={v:summary([x for x in r['rows'] if x['variant']==v]) for v in VARIANTS}:raise ValueError('Summary mismatch')
        if r['negative_summary']!=dict(frame_false_positive_rate=sum(bool(x['predictions']) for x in r['negative_rows'])/48,unmatched_predictions=sum(len(x['predictions']) for x in r['negative_rows'])):raise ValueError('FPR mismatch')
    result=read(OUT/'evaluation/summary.json');verify(result);paths.append(OUT/'evaluation/summary.json')
    if aggregate([records[k] for k in KEYS])!=result['aggregate']['lineage-capped']:raise ValueError('Aggregate mismatch')
    e=read(OUT/'error-review-v1/evidence.json');r=read(OUT/'error-review-v1/review.json');verify(e);verify(r);verify_decisions(e,r['decisions'])
    paths.extend([OUT/'error-review-v1/evidence.json',OUT/'error-review-v1/review.json'])
    tests=['tests.test_lineage_capped_control','tests.test_lineage_capped_results','tests.test_risk_capped_control_review','tests.test_risk_capped_redistribution','tests.test_whole_image_hold_train','tests.test_matched_appearance_training','tests.test_structure_fit','tests.test_structure_fit_integrity']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths += [ROOT/(x.replace('.','/')+'.py') for x in tests]+[ROOT/'docs/lineage-capped-exposure-control-v1.md',ROOT/'docs/results/ml_lineage_capped_control_20260909.md',ROOT/'scripts/vision/evaluate_lineage_capped_control.py',ROOT/'scripts/vision/train_lineage_capped_v2.py']
    uncertain=any(d.get('visual_content')=='unknown' for d in r['decisions'])
    eligible=result['policy_results']['passed'] and not uncertain
    frozen(OUT/'completion.json',dict(status='development_comparison_complete',new_training_units=3,new_evaluation_units=3,
        numerical_gate_passed=result['policy_results']['passed'],development_candidate_eligible=eligible,
        selected_candidate_family='lineage-capped-450' if eligible else None,all_three_seeds_retained=True,
        AI_review_decisions=len(r['decisions']),unresolved_visual_items=uncertain,baseline=baseline,regression_output=t.stderr,
        whole_repository_tested=False,training_admitted=False,promotable=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('COMPARISON_COMPLETE',result['policy_results']['passed'],'PINNED40_PASSED')

if __name__=='__main__':main()
