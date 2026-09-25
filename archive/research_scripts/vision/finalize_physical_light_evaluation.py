"""Close reviewed feedback with explicit competing-assignment evidence."""
from pathlib import Path
from scripts.vision.record_physical_light_error_review import DEST,prior,validate
from scripts.vision.physical_low_light_design import OUT
from scripts.vision.analyze_recovery_paired_calibration import iou

def main():
    dest=OUT/'evaluation-completion.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    deps=[DEST/'evidence.json',DEST/'review.json',DEST/'review-summary.json',OUT/'evaluation/numerical-audit.json',OUT/'evaluation/summary.json',Path(__file__)]
    e,r,rs,a,s=[prior.read(p) for p in deps[:5]]
    for x in (e,r,rs,a,s):prior.verify(x)
    validate(e,r['decisions']);competitions=[]
    for t in rs['loss_details']:
        if not t['miss']['formal_matching_competition']:continue
        family=t['comparison'].split('->')[1]
        ep=OUT/'evaluation'/f"{family}-{t['seed']}.json";p=prior.read(ep);prior.verify(p);deps.append(ep)
        row=next(x for x in p['rows'] if (x['pair_id'],x['variant'])==(t['pair_id'],t['variant']));proof=[]
        for j,pred in enumerate(row['predictions']):
            if pred['class_name']!=t['truth']['class_name'] or iou(pred['bbox_xyxy'],t['truth']['bbox_xyxy'])<.5:continue
            matches=[m for m in row['matches'] if m['prediction_index']==j]
            if len(matches)!=1 or row['truth'][matches[0]['truth_index']]==t['truth']:raise ValueError('Unexplained competition')
            proof.append(dict(prediction=pred,assigned_truth=row['truth'][matches[0]['truth_index']]))
        if not proof:raise ValueError('Missing competition evidence')
        competitions.append(dict(review_id=t['review_decision'],seed=t['seed'],missed_truth=t['truth'],proof=proof))
    if len(competitions)!=a['formal_matching_competitions']:raise ValueError('Competition inventory changed')
    return prior.frozen(dest,dict(status='evaluation_and_AI_review_complete_with_named_gaps_no_candidate',
        review_decisions=len(r['decisions']),pending_content_ids=r['pending_ids'],matching_competitions=competitions,
        planned_assignment_conflicts=s['matching_conflicts'],policy_passed={k:v['passed'] for k,v in s['policy_results'].items()},selected_candidate=None,
        conclusion='Paired physical-lighting tail results only; original and repeated-budget comparisons retained. Content unknowns remain in full metrics. No promotion or sealed test.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':print(main()['status'])
