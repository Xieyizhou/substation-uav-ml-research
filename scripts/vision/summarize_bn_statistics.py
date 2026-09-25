"""Recompute diagnostic scores and preserve an explicit pending-review queue."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_bn_statistics import OUT,SOURCE,prior,paired_truth,score,summary,VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate
from scripts.vision.analyze_visibility_quality_results import paired_change

def main():
    dp=SOURCE/'design.json';p=prior.read(dp);prior.verify(p)
    rp=Path(p['evaluation']['paired_review']);review=prior.read(rp);prior.verify(review)
    pairs,_=paired_truth(review['frames']);truths={(r['pair_id'],r['variant']):(r,t) for r,t in pairs}
    deps=[dp,rp,Path(__file__).resolve()];records={};pending=[];changes={}
    for seed in (7,17,27):
        sp=OUT/f'seed-{seed}.json';r=prior.read(sp);prior.verify(r);deps.append(sp)
        for condition,x in r['outputs'].items():
            for row in x['rows']:
                a,t=truths[row['pair_id'],row['variant']]
                if row!=score(a,t,row['predictions'],row['low_predictions']):raise ValueError('Score changed')
            if x['summary']!={v:summary([y for y in x['rows'] if y['variant']==v]) for v in VARIANTS}:raise ValueError('Summary changed')
            records[condition,seed]=x
            if '_' in condition:
                for row in x['negative_rows']:
                    for j,pred in enumerate(row['predictions']):pending.append(dict(kind='FP',seed=seed,condition=condition,image_sha256=row['image_sha256'],view_id=row['view_id'],variant=row['variant'],prediction_index=j,prediction=pred,review_status='pending'))
        for a,b in (('R','R_Bstats'),('B','B_Rstats')):
            left,right=r['outputs'][a],r['outputs'][b]
            changes[f'{a}->{b}:{seed}']=paired_change(left['rows'],right['rows'])
            lookup={(z['pair_id'],z['variant']):z for z in left['rows']}
            for row in right['rows']:
                old=lookup[row['pair_id'],row['variant']]
                oldhit={ (old['truth'][m['truth_index']]['class_name'],tuple(old['truth'][m['truth_index']]['bbox_xyxy'])) for m in old['matches']}
                hit={m['truth_index'] for m in row['matches']}
                for j,t in enumerate(row['truth']):
                    if row['variant'] in ('original','lighting') and j not in hit and (t['class_name'],tuple(t['bbox_xyxy'])) in oldhit:
                        pending.append(dict(kind='LOSS',seed=seed,condition=b,pair_id=row['pair_id'],variant=row['variant'],image_sha256=row['image_sha256'],truth=t,miss=next(m for m in row['misses'] if m['truth_index']==j),review_status='pending'))
    groups={c:aggregate([records[c,s] for s in (7,17,27)]) for c in ('B','R','R_Bstats','B_Rstats')}
    result=prior.frozen(OUT/'summary.json',dict(status='numerical_complete_explicit_review_pending',aggregate=groups,paired_changes=changes,
        review_queue=pending,review_counts=dict(Counter(x['kind'] for x in pending)),selected_candidate=None,diagnostic_only=True,
        inputs={str(d):prior.file_sha256(d) for d in deps}))
    for c,g in groups.items():print(c,{v:round(g[v]['instance_recall']['mean'],6) for v in ('original','lighting')},g['no_target']['frame_false_positive_rate']['mean'])
    print('REVIEW',result['review_counts'])

if __name__=='__main__':main()
