"""Strict completion: recomputed evaluation, explicit review, actual augmentation."""
import subprocess,sys
from pathlib import Path
from collections import Counter
from scripts.vision.train_brightness_transfer import OUT,ROOT,KEYS,pretrain,complete,read,verify,frozen,file_sha256
from scripts.vision.brightness_transfer_runtime import factor
from scripts.vision.check_lineage_training_fit import OUT as FIT,truth_for,scoring
from scripts.vision.run_fixed_budget_diagnosis import paired_truth,score,summary,VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.prepare_whole_image_hold import baseline_verify

def validate_review(e,ds):
    expected={f"{r['event_id']}:{i}":(r,i) for r in e['events'] for i in range(len(r['predictions']) if r['kind']=='FP' else 1)}
    if len(ds)!=len(expected) or {d['review_id'] for d in ds}!=set(expected):raise ValueError('Missing/duplicate visual decisions')
    for d in ds:
        r,i=expected[d['review_id']]
        if d['evidence_sha256']!=r['evidence_sha256'] or d['image_sha256']!=r['source']['image_sha256'] or d['crop_sha256']!=r['crops'][i]['sha256']:raise ValueError('Stale visual evidence')
        if d['review_nature']!='AI辅助审核' or not d['reviewed_at'] or not d['reason'] or d['pixel_visibility_certified'] is not False:raise ValueError('Unsupported review')
        if r['kind']=='FP' and d['prediction']!=r['predictions'][i]:raise ValueError('Prediction changed')
        if r['kind']=='LOSS' and d['truth']!=r['truth']:raise ValueError('Truth changed')

def check_factors(p):
    for seed in (7,17,27):
        off=f'noaug-450-{seed}';on=f'brightness-450-{seed}'
        if p['schedules'][off]!=p['schedules'][on] or len(p['schedules'][on])!=2700:raise ValueError('Different exposure sequence')
        if p['brightness_factors'][off]!=[1.]*2700 or p['brightness_factors'][on]!=[factor(seed,m,i) for i,m in enumerate(p['schedules'][on])]:raise ValueError('Gain sequence not reproducible')

def main():
    p=pretrain();check_factors(p);paths=[OUT/'protocol.json',OUT/'training-gate.json',Path(__file__)]
    pairs,_=paired_truth(read(p['evaluation']['paired_review'])['frames']);truth={(r['view_id'],r['variant']):(r,t) for r,t in pairs}
    negatives={(r['view_id'],r['variant']):r for r in read(p['evaluation']['negative_review'])['frames']};records=[];fit_metrics={}
    fp=read(FIT/'protocol.json');verify(fp);fit_idx={r['member_id']:r for r in fp['rows']};fit_details=[]
    for key in KEYS:
        complete(key,p);path=OUT/'evaluation'/f'{key}.json';r=read(path);verify(r);records.append(r);paths.extend([path,OUT/'training'/key/'brightness-receipt.json'])
        if len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete/conflicted evaluation')
        if {(x['view_id'],x['variant']) for x in r['rows']}!=set(truth) or {(x['view_id'],x['variant']) for x in r['negative_rows']}!=set(negatives):raise ValueError('Evaluation membership drift')
        for x in r['rows']:
            source,t=truth[x['view_id'],x['variant']]
            if x!=score(source,t,x['predictions'],x['low_predictions']):raise ValueError('Matching drift')
        if r['summary']!={v:summary([x for x in r['rows'] if x['variant']==v]) for v in VARIANTS}:raise ValueError('Summary drift')
        for x in r['negative_rows']:
            if x['image_sha256']!=negatives[x['view_id'],x['variant']]['image_sha256'] or x['frame_has_prediction']!=bool(x['predictions']):raise ValueError('FPR row drift')
        if r['negative_summary']!=dict(frame_false_positive_rate=sum(bool(x['predictions']) for x in r['negative_rows'])/48,unmatched_predictions=sum(len(x['predictions']) for x in r['negative_rows'])):raise ValueError('FPR drift')
        fitpath=OUT/'fit'/f'{key}.json';f=read(fitpath);verify(f);paths.append(fitpath)
        if len(f['rows'])!=47 or {x['member_id'] for x in f['rows']}!=set(fit_idx):raise ValueError('Fit subset incomplete')
        counts=Counter(p['schedules'][key]);totals=Counter();hits=Counter();clear=Counter();clearn=Counter()
        for x in f['rows']:
            sc=x['scoring'];full=truth_for(fit_idx[x['member_id']])
            if x['exposures']!=counts[x['member_id']] or sc!=scoring(full,sc['predictions'],sc['low_predictions']):raise ValueError('Fit scoring/exposure drift')
            if x['exposures']>0:totals.update(t['class_name'] for t in full);hits.update(t['class_name'] for t in sc['matches'])
            for t in fp['targets']:
                d=t['review']
                if d['member_id']!=x['member_id']:continue
                i=d['truth']['label_line_index'];hit=any(z['truth_index']==i for z in sc['matches']);cls=d['truth']['class_name']
                fit_details.append(dict(cell=key,event_id=d['event_id'],class_name=cls,clear_primary=t['clear_primary'],actual_exposures=x['exposures'],hit=hit,miss=next((m for m in sc['misses'] if m['truth_index']==i),None)))
                if t['clear_primary'] and x['exposures']>0:clearn[cls]+=1;clear[cls]+=int(hit)
        fit_metrics[key]=dict(full={c:dict(hits=hits[c],instances=n,recall=hits[c]/n) for c,n in totals.items()},clear={c:dict(hits=clear[c],instances=n,recall=clear[c]/n) for c,n in clearn.items()})
    result=read(OUT/'evaluation/summary.json');verify(result)
    if aggregate(records)!=result['aggregate']['brightness']:raise ValueError('Aggregate drift')
    from scripts.vision.exposure_order_retention import PRIOR
    refs=[read(PRIOR/f'evaluation-retained_reference-450-{s}.json') for s in (7,17,27)];hist=read(p['evaluation']['historical_reference'])
    for r in refs+[hist]:verify(r)
    if policy_checks(aggregate(records),aggregate(refs),hist['historical_A'],p)!=result['policy_results']:raise ValueError('Gate drift')
    e=read(OUT/'error-review/evidence.json');review=read(OUT/'error-review/review.json');verify(e);verify(review);validate_review(e,review['decisions'])
    fe=read(OUT/'fit-review/evidence.json');fr=read(OUT/'fit-review/review.json');verify(fe);verify(fr)
    expected={f"{d['cell'].split('-')[-1]}:{d['event_id']}" for d in fit_details if d['clear_primary'] and not d['hit']}
    if len(fe['events'])!=len(expected) or {r['review_id'] for r in fe['events']}!=expected or len(fr['decisions'])!=len(expected) or {r['review_id'] for r in fr['decisions']}!=expected:raise ValueError('Clear fit review incomplete/duplicate')
    for d in fr['decisions']:
        r=next(x for x in fe['events'] if x['review_id']==d['review_id'])
        if d['evidence_sha256']!=r['evidence_sha256'] or d['image_sha256']!=r['source']['image_sha256'] or d['truth']!=r['truth'] or not d['reason'] or d['review_nature']!='AI辅助审核' or d['pixel_visibility_certified'] is not False:raise ValueError('Fit review invalid')
    paths.extend([OUT/'fit-review/evidence.json',OUT/'fit-review/review.json'])
    paths.extend([OUT/'evaluation/summary.json',OUT/'error-review/evidence.json',OUT/'error-review/review.json',FIT/'protocol.json',ROOT/'docs/brightness-transfer-control-v1.md',ROOT/'docs/results/ml_brightness_transfer_20260910.md'])
    tests=['tests.test_brightness_transfer','tests.test_brightness_audit','tests.test_lineage_training_fit','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_capped_results']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    paths.extend(ROOT/(x.replace('.','/')+'.py') for x in tests)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Pinned baseline failure')
    unknown=any(d.get('visual_content')=='unknown' for d in review['decisions'])
    passed=result['policy_results']['passed'] and not unknown
    frozen(OUT/'completion.json',dict(status='brightness_control_development_complete',new_training_units=3,reused_baselines=3,formal_and_low_evaluation_units=3,training_fit_units=3,fit_metrics=fit_metrics,fit_details=fit_details,numerical_gate_passed=result['policy_results']['passed'],unresolved_visual_items=unknown,selected_candidate='brightness-450' if passed else None,baseline=baseline,regression_output=t.stderr,whole_repository_tested=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('COMPLETE_BRIGHTNESS_CONTROL',passed,flush=True)

if __name__=='__main__':main()
