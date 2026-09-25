"""Matched T/Q member fit and frozen historical loss-cohort diagnosis."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_material_fixed_sequence_lr_fit import OUT,PREVIOUS,TRAIN,prior,freeze,validate
from scripts.vision.diagnose_material_retention_transfer import validate as validate_old

def identity(t):return (t['object_id'],t['class_name'],tuple(t['bbox_xyxy']))

def match_truth(before,after):
    a={identity(t):i for i,t in enumerate(before)};b={identity(t):i for i,t in enumerate(after)}
    if len(a)!=len(before) or len(b)!=len(after) or set(a)!=set(b):raise ValueError('Ambiguous/mismatched truth')
    return [(a[k],b[k]) for k in a]

def main():
    p=freeze();oldp=prior.read(PREVIOUS/'protocol.json');prior.verify(oldp)
    trajectory=TRAIN.parent/'material-learning-trajectory-v1/analysis-v1/summary-v2.json'
    ts=prior.read(trajectory);prior.verify(ts)
    cohort={(t['cell'],t['member_id'],identity(t['truth'])) for t in ts['trajectories']
        if t['mode']=='ema' and t['planned_target'] and t['exposure_steps'] and t['classification']=='previously_hit_endpoint_miss'}
    if len(cohort)!=13:raise ValueError('Historical cohort drift')
    paths=[OUT/'protocol.json',PREVIOUS/'protocol.json',trajectory,Path(__file__).resolve()];details=[];counts=[]
    for seed in (7,17,27):
        nk,ok=f'Q-{seed}',f'T-{seed}'
        np,op=OUT/(nk+'.json'),PREVIOUS/(ok+'.json')
        new,old=prior.read(np),prior.read(op);validate(new,nk,p);validate_old(old,ok,oldp);paths.extend((np,op))
        if p['actual_exposures'][nk]!=oldp['actual_exposures'][ok]:raise ValueError('Unequal exposure multiset')
        olds={r['member_id']:r for r in old['rows']};news={r['member_id']:r for r in new['rows']}
        for m in p['members']:
            mid=m['member_id'];a,b=olds[mid],news[mid]
            if a['image_sha256']!=b['image_sha256']:raise ValueError('Different image')
            ah={x['truth_index'] for x in a['matches']};bh={x['truth_index'] for x in b['matches']}
            for i,j in match_truth(a['truth'],b['truth']):
                t=b['truth'][j];before=i in ah;after=j in bh
                details.append(dict(seed=seed,member_id=mid,source_id=m['source_id'],condition=m['condition'],truth=t,
                    image_sha256=m['image_sha256'],planned_target=t['object_id']==m['target'],
                    exposures=p['actual_exposures'][nk].get(mid,0),before_hit=before,after_hit=after,
                    state='retained_hit' if before and after else 'loss' if before else 'gain' if after else 'retained_miss',
                    historical_loss_cohort=(ok,mid,identity(t)) in cohort,
                    before_miss=next((x for x in a['misses'] if x['truth_index']==i),None),
                    after_miss=next((x for x in b['misses'] if x['truth_index']==j),None)))
        for condition in sorted({m['condition'] for m in p['members']}):
            group=[x for x in details if x['seed']==seed and x['condition']==condition]
            target=[x for x in group if x['planned_target']];exposed=[x for x in target if x['exposures']]
            counts.append(dict(seed=seed,condition=condition,target_count=len(target),
                T_target_hits=sum(x['before_hit'] for x in target),Q_target_hits=sum(x['after_hit'] for x in target),
                full_truth_count=len(group),T_full_hits=sum(x['before_hit'] for x in group),Q_full_hits=sum(x['after_hit'] for x in group),
                actual_exposed_target_count=len(exposed),T_exposed_hits=sum(x['before_hit'] for x in exposed),Q_exposed_hits=sum(x['after_hit'] for x in exposed)))
    exposed=[x for x in details if x['planned_target'] and x['exposures']]
    cohort_rows=[x for x in details if x['historical_loss_cohort']]
    if len(cohort_rows)!=13:raise ValueError('Incomplete matched cohort')
    result=dict(status='matched_member_fit_diagnosis_complete',counts=counts,details=details,
        actual_exposed_target_transitions=dict(Counter(x['state'] for x in exposed)),historical_cohort=cohort_rows,
        historical_recovered=sum(x['after_hit'] for x in cohort_rows),selected_candidate=None,
        limits=['Training fit on original RGB, not augmented tensors or independent generalization.',
                'All full-image labels retained; zero exact member exposure does not certify unseen same-source pixels.',
                'Historical cohort full-precision trajectory versus terminal half serialization is explicitly distinguished.',
                'Recovery at endpoint does not establish a new temporal forgetting mechanism.'],
        inputs={str(x):prior.file_sha256(x) for x in paths})
    return prior.frozen(OUT/'summary.json',result)

if __name__=='__main__':
    r=main();print(r['status']);print(r['actual_exposed_target_transitions']);print('cohort recovered',r['historical_recovered'],'/13')
