"""Recompute scores, confirm pairing and annotate matching competition."""
from collections import Counter
from pathlib import Path
from scripts.vision.evaluate_closed_budget import OUT,KEYS,prior,contract,complete,validate_record,verify_binding,paired_truth,score,summary,VARIANTS,direct_checks
from scripts.vision.record_closed_budget_review import DEST,validate
from scripts.vision.analyze_recovery_paired_calibration import iou

def main():
    dest=OUT/'evaluation/audit.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    _,p,_=contract(KEYS[0]);rp=Path(p['evaluation']['paired_review']);review=prior.read(rp);prior.verify(review)
    pairs,_=paired_truth(review['frames']);source={(r['pair_id'],r['variant']):(r,t) for r,t in pairs}
    paths=[rp,OUT/'evaluation/summary.json',DEST/'evidence.json',DEST/'review.json',DEST/'review-summary.json',Path(__file__).resolve()]
    s=prior.read(paths[1]);e=prior.read(paths[2]);r=prior.read(paths[3]);rs=prior.read(paths[4])
    for x in (s,e,r,rs):prior.verify(x)
    validate(e,r['decisions']);records={}
    for key in KEYS:
        complete(key);ep=OUT/'evaluation'/f'{key}.json';x=prior.read(ep);validate_record(x,key);verify_binding(x,key);paths.append(ep);records[key]=x
        for row in x['rows']:
            original,truth=source[row['pair_id'],row['variant']]
            if score(original,truth,row['predictions'],row['low_predictions'])!=row:raise ValueError('Score/identity changed')
        if x['summary']!={v:summary([row for row in x['rows'] if row['variant']==v]) for v in VARIANTS}:raise ValueError('Summary not reproducible')
        negative=dict(frame_false_positive_rate=sum(bool(row['predictions']) for row in x['negative_rows'])/48,unmatched_predictions=sum(len(row['predictions']) for row in x['negative_rows']))
        if negative!=x['negative_summary']:raise ValueError('Negative summary drift')
    prefix={}
    for seed in (7,17,27):
        exposures=[]
        for budget in (450,900):
            cp=OUT/'training'/f'B{budget}-{seed}'/'completion.json';c=prior.read(cp);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x);exposures.append(x);paths.extend([cp,xp])
        a,b=exposures
        exact=[r['loss_items'] for r in a['step_records']]==[r['loss_items'] for r in b['step_records'][:450]]
        if not exact or b['actual']!=a['actual']*2:raise ValueError('Actual controlled prefix drift')
        prefix[str(seed)]=dict(first_450_losses_exact=True,actual_members_exact_repeat=True,first_450_tensor_records_exact=b['batch_records'][:450]==a['batch_records'])
    competitions=[]
    for t in rs['loss_details']:
        if not t['miss']['formal_matching_competition']:continue
        row=next(x for x in records[f"B900-{t['seed']}"]['rows'] if x['pair_id']==t['pair_id'] and x['variant']==t['variant'])
        competing=[]
        for j,pred in enumerate(row['predictions']):
            overlap=iou(pred['bbox_xyxy'],t['truth']['bbox_xyxy'])
            if pred['class_name']!=t['truth']['class_name'] or overlap<.5:continue
            assigned=[m for m in row['matches'] if m['prediction_index']==j]
            if len(assigned)!=1 or row['truth'][assigned[0]['truth_index']]==t['truth']:raise ValueError('Unexplained formal matching conflict')
            competing.append(dict(prediction=pred,iou_to_missed=overlap,assigned_match=assigned[0],assigned_truth=row['truth'][assigned[0]['truth_index']]))
        if not competing:raise ValueError('Missing competition proof')
        competitions.append(dict(review_decision=t['review_decision'],seed=t['seed'],variant=t['variant'],truth=t['truth'],evidence=competing,
            original_reason=t['miss']['reason'],interpretation='Formal same-class prediction assigned to overlapping other instance; not simple absence of candidate.'))
    return prior.frozen(dest,dict(status='evaluation_and_explicit_review_complete_with_named_gaps_no_candidate',cells=list(KEYS),predictions_recomputed_without_inference=True,
        prefix_checks=prefix,matching_competitions=competitions,plan_full_assignment_conflicts=s['matching_conflicts'],
        noncompetition_loss_reasons=dict(Counter(t['miss']['reason'] for t in rs['loss_details'] if not t['miss']['formal_matching_competition'])),
        within_experiment_retention=direct_checks(s['aggregate']['B900'],s['aggregate']['B450']),review_decisions=len(r['decisions']),pending_content_ids=r['pending_ids'],
        historical_reference_label_note='Legacy policy key same_budget_R refers to frozen retained_reference-450 for BOTH arms; it is not a newly trained 900-step R family.',
        selected_candidate=None,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=main();print(r['status'],r['review_decisions'],'decisions',len(r['pending_content_ids']),'gaps',len(r['matching_competitions']),'competitions')
