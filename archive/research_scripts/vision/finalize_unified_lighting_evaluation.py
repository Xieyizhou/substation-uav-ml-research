"""Read explicit decisions, verify outputs and write bounded evaluation conclusion."""
import subprocess,sys
from collections import Counter
from pathlib import Path
from scripts.vision.evaluate_unified_lighting import OUT,KEYS,ready,complete,prior,validate_record,baseline_verify
from scripts.vision.record_unified_lighting_errors import validate

def audit_review(e,r):
    validate(e,r['decisions'])
    items={x['event_id']:x for x in e['events']}
    for d in r['decisions']:
        eid,j=d['decision_id'].split(':');event=items[eid];j=int(j)
        if event['kind']=='FP':
            if d['prediction']!=event['predictions'][j]:raise ValueError('Reviewed prediction changed')
        elif d['truth']!=event['loss']['truth']:raise ValueError('Reviewed truth changed')
    return [d['decision_id'] for d in r['decisions'] if d['status']=='pending']

def candidate_allowed(summary,pending):
    return not pending and summary['matching_conflicts']==0 and summary['policy_results']['L-physical']['passed'] and summary['direct_retention']['passed']

def main():
    dest=OUT/'evaluation/completion.json'
    if dest.exists():prior.verify(prior.read(dest));print('VERIFIED_EXISTING');return
    p=ready();sp=OUT/'evaluation/summary.json';ep=OUT/'error-review/evidence.json';rp=OUT/'error-review/review.json'
    s,e,r=map(prior.read,(sp,ep,rp))
    for x in (s,e,r):prior.verify(x)
    pending=audit_review(e,r);paths=[sp,ep,rp,OUT/'protocol.json',OUT/'training/completion.json',Path(__file__).resolve()]
    records={}
    for k in KEYS:
        complete(k,p);path=OUT/'evaluation'/f'{k}.json';records[k]=prior.read(path);validate_record(records[k],k);paths += [path,OUT/'training'/k/'tensor-completion.json']
    suites=['tests.test_unified_lighting_evaluation','tests.test_unified_lighting_design','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation','tests.test_structure_fit','tests.test_structure_fit_integrity']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode:raise ValueError('Scoped regressions failed: '+t.stdout+t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in suites]
    # Comparison occurrences are not unique misses; deduplicate reference comparisons.
    misses={}
    for event in e['events']:
        if event['kind']=='LOSS':
            for x in event['loss']['events']:misses[(event['event_id'],x['cell'])]=x['miss']['reason']
    result=prior.frozen(dest,dict(status='evaluation_complete_with_named_gaps_no_candidate' if pending else 'evaluation_complete',
        selected_candidate='L-physical-three-seed-development-only' if candidate_allowed(s,pending) else None,
        numerical_policy_passed={a:v['passed'] for a,v in s['policy_results'].items()},direct_retention_passed=s['direct_retention']['passed'],
        matching_conflicts=s['matching_conflicts'],reviewed_predictions=sum(len(x['predictions']) for x in e['events']),
        negative_unique_images=sum(x['kind']=='FP' for x in e['events']),loss_unique_boxes=sum(x['kind']=='LOSS' for x in e['events']),
        named_gaps=pending,unique_cell_loss_reasons=dict(Counter(misses.values())),
        comparisons='L-physical versus same-seed R-clean; both versus previous brightness additionally retained in transitions. Repeated predictions are not new scenes.',
        regression_output=t.stdout+t.stderr,baseline=baseline,whole_repository_tests_claimed=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(result['status'],result['unique_cell_loss_reasons'],flush=True)

if __name__=='__main__':main()
