"""Recompute results and require current explicit visual decisions."""
import subprocess
import sys
from collections import Counter
from pathlib import Path
from scripts.vision.brightness_lr_retention import OUT, SOURCE, ROOT, KEYS, TESTS, ready, complete, read, verify, frozen, file_sha256, baseline_verify
from scripts.vision.audit_brightness_transfer import validate_review, check_factors
from scripts.vision.check_lineage_training_fit import OUT as FIT, truth_for, scoring
from scripts.vision.run_fixed_budget_diagnosis import paired_truth, score, summary, VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks


def validate_current_losses(e, records, baselines):
    transitions=[];wanted=[]
    for seed,new,old in zip((7,17,27),records,baselines,strict=True):
        idx={(r['view_id'],r['variant']):r for r in old['rows']}
        for r in new['rows']:
            a=idx[r['view_id'],r['variant']]
            if r['truth']!=a['truth'] or r['image_sha256']!=a['image_sha256']:raise ValueError('Paired identity drift')
            ah={m['truth_index'] for m in a['matches']};bh={m['truth_index'] for m in r['matches']}
            for i,t in enumerate(r['truth']):
                state='persistent_hit' if i in ah and i in bh else 'gain' if i in bh else 'loss' if i in ah else 'persistent_miss'
                transitions.append(dict(seed=seed,pair_id=r['pair_id'],variant=r['variant'],truth=t,state=state))
                if state=='loss' and r['variant'] in ('original','lighting') and t['class_name'] in ('reactor','capacitor_bank'):
                    wanted.append((seed,r['image_sha256'],t,next(m for m in r['misses'] if m['truth_index']==i)))
    actual=[(x['seed'],r['source']['image_sha256'],r['truth'],x['miss']) for r in e['events'] if r['kind']=='LOSS' for x in r['events']]
    if sorted(wanted,key=str)!=sorted(actual,key=str) or e['all_transitions']!=transitions:
        raise ValueError('Current loss/persistent/gain review coverage drift')


def compute(p):
    pairs,_=paired_truth(read(p['evaluation']['paired_review'])['frames'])
    truths={(r['view_id'],r['variant']):(r,t) for r,t in pairs}
    negatives={(r['view_id'],r['variant']):r for r in read(p['evaluation']['negative_review'])['frames']}
    fit=read(FIT/'protocol.json');verify(fit);idx={r['member_id']:r for r in fit['rows']}
    records=[];metrics={};details=[]
    for key in KEYS:
        complete(key,p);r=read(OUT/'evaluation'/f'{key}.json');verify(r);records.append(r)
        if len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete/conflicted evaluation')
        if {(x['view_id'],x['variant']) for x in r['rows']}!=set(truths) or {(x['view_id'],x['variant']) for x in r['negative_rows']}!=set(negatives):raise ValueError('Development membership drift')
        for x in r['rows']:
            source,t=truths[x['view_id'],x['variant']]
            if x!=score(source,t,x['predictions'],x['low_predictions']):raise ValueError('Matching drift')
        if r['summary']!={v:summary([x for x in r['rows'] if x['variant']==v]) for v in VARIANTS}:raise ValueError('Summary drift')
        for x in r['negative_rows']:
            if x['image_sha256']!=negatives[x['view_id'],x['variant']]['image_sha256'] or x['frame_has_prediction']!=bool(x['predictions']):raise ValueError('Negative row drift')
        if r['negative_summary']!=dict(frame_false_positive_rate=sum(bool(x['predictions']) for x in r['negative_rows'])/48,unmatched_predictions=sum(len(x['predictions']) for x in r['negative_rows'])):raise ValueError('FPR drift')
        f=read(OUT/'fit'/f'{key}.json');verify(f)
        if len(f['rows'])!=47 or {x['member_id'] for x in f['rows']}!=set(idx):raise ValueError('Fit membership drift')
        totals=Counter();hits=Counter();clear=Counter();clearn=Counter();counts=Counter(p['schedules'][key])
        for x in f['rows']:
            sc=x['scoring'];full=truth_for(idx[x['member_id']])
            if sc!=scoring(full,sc['predictions'],sc['low_predictions']) or x['exposures']!=counts[x['member_id']]:raise ValueError('Fit scoring/exposure drift')
            if x['exposures']>0:totals.update(t['class_name'] for t in full);hits.update(t['class_name'] for t in sc['matches'])
            for t in fit['targets']:
                d=t['review']
                if d['member_id']!=x['member_id']:continue
                i=d['truth']['label_line_index'];hit=any(z['truth_index']==i for z in sc['matches']);cls=d['truth']['class_name']
                details.append(dict(cell=key,event_id=d['event_id'],class_name=cls,clear_primary=t['clear_primary'],actual_exposures=x['exposures'],hit=hit,miss=next((m for m in sc['misses'] if m['truth_index']==i),None)))
                if t['clear_primary'] and x['exposures']>0:clearn[cls]+=1;clear[cls]+=int(hit)
        metrics[key]=dict(full={c:dict(hits=hits[c],instances=n,recall=hits[c]/n) for c,n in totals.items()},clear={c:dict(hits=clear[c],instances=n,recall=clear[c]/n) for c,n in clearn.items()})
    result=read(OUT/'evaluation/summary.json');verify(result)
    if aggregate(records)!=result['aggregate']['lr0005']:raise ValueError('Aggregate drift')
    for field,group in [('baselines','lr001'),('unaugmented_context','noaug_context')]:
        old=[read(p[field][str(s)]['evaluation']) for s in (7,17,27)]
        for r in old:verify(r)
        if aggregate(old)!=result['aggregate'][group]:raise ValueError('Reference aggregate drift')
    from scripts.vision.exposure_order_retention import PRIOR
    refs=[read(PRIOR/f'evaluation-retained_reference-450-{s}.json') for s in (7,17,27)];hist=read(p['evaluation']['historical_reference'])
    for r in refs+[hist]:verify(r)
    if policy_checks(aggregate(records),aggregate(refs),hist['historical_A'],p)!=result['policy_results']:raise ValueError('Gate drift')
    return result,metrics,details


def main():
    p=ready();check_factors(p);result,metrics,details=compute(p)
    paths=[OUT/'protocol.json',OUT/'ready.json',OUT/'evidence-generation.json',Path(__file__),
           ROOT/'scripts/vision/audit_brightness_transfer.py',ROOT/'scripts/vision/evaluate_brightness_lr_retention.py',
           ROOT/'docs/brightness-lr-retention-v1.md',ROOT/'docs/results/ml_brightness_lr_retention_20260910.md']
    import torch,ultralytics
    if p['environment']!=dict(torch=torch.__version__,ultralytics=ultralytics.__version__):raise ValueError('Current environment differs from control')
    for root in (OUT,SOURCE):
        for key in KEYS:
            logs=[]
            for lp in (root/'workers'/key).glob('attempt-*/log.txt'):
                if any(f'Ultralytics {ultralytics.__version__}' in line and f'torch-{torch.__version__}' in line and 'CPU (Apple M2 Pro)' in line for line in lp.read_text().splitlines()):logs.append(lp)
            if len(logs)!=1:raise ValueError('Ambiguous historical/new runtime environment')
            paths.extend(logs)
    e=read(OUT/'error-review/evidence.json');r=read(OUT/'error-review/review.json');verify(e);verify(r);validate_review(e,r['decisions'])
    validate_current_losses(e,[read(OUT/'evaluation'/f'{k}.json') for k in KEYS],
                            [read(p['baselines'][str(s)]['evaluation']) for s in (7,17,27)])
    # Ensure rendering covered every current FP (seed repetitions retained, not independent images).
    wanted=[]
    for key in KEYS:
        for row in read(OUT/'evaluation'/f'{key}.json')['negative_rows']:
            wanted.extend((int(key.split('-')[-1]),row['image_sha256'],i,x) for i,x in enumerate(row['predictions']))
    actual=[]
    for event in e['events']:
        if event['kind']=='FP':
            for x in event['predictions']:
                actual.append((x['seed'],event['source']['image_sha256'],x['prediction_index'],{k:v for k,v in x.items() if k not in ('seed','prediction_index')}))
    if sorted(wanted,key=str)!=sorted(actual,key=str):raise ValueError('FP review coverage drift')
    fe=read(OUT/'fit-review/evidence.json');fr=read(OUT/'fit-review/review.json');verify(fe);verify(fr)
    expected={f"{d['cell'].split('-')[-1]}:{d['event_id']}" for d in details if d['clear_primary'] and not d['hit']}
    if len(fe['events'])!=len(expected) or {x['review_id'] for x in fe['events']}!=expected or len(fr['decisions'])!=len(expected) or {x['review_id'] for x in fr['decisions']}!=expected:raise ValueError('Clear fit review missing/duplicate')
    for d in fr['decisions']:
        x=next(x for x in fe['events'] if x['review_id']==d['review_id'])
        if d['evidence_sha256']!=x['evidence_sha256'] or d['image_sha256']!=x['source']['image_sha256'] or d['truth']!=x['truth'] or not d['reason'] or not d['reviewed_at'] or d['review_nature']!='AI辅助审核' or d['pixel_visibility_certified'] is not False:raise ValueError('Fit review invalid')
    paths += [OUT/d/f for d in ('error-review','fit-review') for f in ('evidence.json','review.json')]
    paths += [OUT/'evaluation/summary.json',FIT/'protocol.json']
    paths += [OUT/d/f'{k}.json' for d in ('evaluation','fit') for k in KEYS]
    paths += [OUT/'training'/k/'brightness-receipt.json' for k in KEYS]
    tests=TESTS+['tests.test_brightness_lr_audit','tests.test_brightness_lr_process','tests.test_lineage_training_fit','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_capped_results']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    paths += [ROOT/(x.replace('.','/')+'.py') for x in tests]
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Pinned baseline failure')
    unknown=any(x.get('visual_content')=='unknown' for x in r['decisions']+fr['decisions'])
    passed=result['policy_results']['passed'] and not unknown
    frozen(OUT/'completion.json',dict(status='lr_control_development_complete',new_training_units=3,reused_baselines=3,formal_and_low_evaluation_units=3,training_fit_units=3,
        fit_metrics=metrics,fit_details=details,numerical_gate_passed=result['policy_results']['passed'],unresolved_visual_items=unknown,
        selected_candidate='brightness-lr0005-450' if passed else None,baseline=baseline,regression_output=t.stderr,whole_repository_tested=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('COMPLETE_LR_CONTROL',passed,flush=True)


if __name__=='__main__':main()
