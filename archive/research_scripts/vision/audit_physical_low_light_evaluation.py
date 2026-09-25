"""Recompute numerical results and verify actual single-factor comparison."""
from collections import Counter
from pathlib import Path
from scripts.vision.evaluate_physical_low_light import OUT,KEYS,prior,complete,contract,validate_record,verify_binding,paired_truth,score,summary,VARIANTS
from scripts.vision.closed_gamma_design import SOURCE

def main():
    dest=OUT/'evaluation/numerical-audit.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    _,p,_=contract(KEYS[0]);rp=Path(p['evaluation']['paired_review']);review=prior.read(rp);prior.verify(review)
    pairs,_=paired_truth(review['frames']);truths={(r['pair_id'],r['variant']):(r,t) for r,t in pairs}
    deps=[rp,OUT/'evaluation/summary.json',OUT/'evaluation-runner/completion.json',Path(__file__).resolve(),OUT/'evaluation/error-review-v1/evidence.json']
    for key in KEYS:
        complete(key);ep=OUT/'evaluation'/f'{key}.json';r=prior.read(ep);validate_record(r,key);verify_binding(r,key);deps.append(ep)
        for row in r['rows']:
            a,t=truths[row['pair_id'],row['variant']]
            if score(a,t,row['predictions'],row['low_predictions'])!=row:raise ValueError('Score recomputation differs')
        if r['summary']!={v:summary([x for x in r['rows'] if x['variant']==v]) for v in VARIANTS}:raise ValueError('Summary differs')
        if r['negative_summary']!=dict(frame_false_positive_rate=sum(bool(x['predictions']) for x in r['negative_rows'])/48,unmatched_predictions=sum(len(x['predictions']) for x in r['negative_rows'])):raise ValueError('Negative metric differs')
        ref='B900-'+key.split('-')[-1];oldp=SOURCE/'evaluation'/f'{ref}.json';old=prior.read(oldp);validate_record(old,ref);deps.append(oldp)
        if r['effective_cpu_threads']!=old['effective_cpu_threads'] or r['effective_cpu_threads']!=4:raise ValueError('CPU mismatch')
        exposures=[]
        for root,cell in ((OUT,key),(SOURCE,ref)):
            cp=root/'training'/cell/'completion.json';c=prior.read(cp);prior.verify(c);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x);exposures.append(x);deps.extend([cp,xp])
        a,b=exposures
        if a['actual'][:5400]!=b['actual'] or a['brightness_log'][:5400]!=b['brightness_log'] or a['batch_records'][:900]!=b['batch_records']:raise ValueError('Actual base prefix drift')
    for seed in (7,17,27):
        xs=[]
        for family in ('R1000','L1000'):
            c=prior.read(OUT/'training'/f'{family}-{seed}'/'completion.json');xs.append(prior.read(c['exposure_path']))
        a,b=xs
        if sum(x!=y for x,y in zip(a['actual'],b['actual'],strict=True))!=300:raise ValueError('Actual treatment dose drift')
        for x,y in zip(a['batch_records'],b['batch_records'],strict=True):
            if x['full_supervision']!=y['full_supervision']:raise ValueError('Paired actual label drift')
            if x['members']==y['members'] and x!=y:raise ValueError('Common actual tensor changed')
    e=prior.read(deps[4]);prior.verify(e)
    for d in deps:
        if d.suffix=='.json':prior.verify(prior.read(d))
    return prior.frozen(dest,dict(status='numerical_evaluation_verified_explicit_error_review_pending',
        score_recomputed=True,actual_base_prefix_exact=True,paired_full_supervision_equal=True,common_actual_tensors_equal=True,
        transitions=dict(Counter(x['state'] for x in e['transitions'])),loss_reasons=dict(Counter(x['miss']['reason'] for x in e['transitions'] if x['state']=='loss')),
        formal_matching_competitions=sum(bool(x['miss']['formal_matching_competition']) for x in e['transitions'] if x['state']=='loss'),
        pending_error_cards=len(e['events']),pending_fp_predictions=sum(len(x['events']) for x in e['events'] if x['kind']=='FP'),pending_loss_targets=sum(x['kind']=='LOSS' for x in e['events']),
        selected_candidate=None,review_complete=False,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print(main())
