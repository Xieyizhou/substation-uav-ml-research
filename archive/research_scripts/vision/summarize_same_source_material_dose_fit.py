"""Matched T/Q member fit and frozen historical loss-cohort diagnosis."""
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.diagnose_same_source_material_dose_fit import OUT,PREVIOUS,TRAIN,prior,freeze,validate
from scripts.vision.diagnose_material_retention_transfer import validate as validate_old

def identity(t):return (t['object_id'],t['class_name'],tuple(t['bbox_xyxy']))

def match_truth(before,after):
    a={identity(t):i for i,t in enumerate(before)};b={identity(t):i for i,t in enumerate(after)}
    if len(a)!=len(before) or len(b)!=len(after) or set(a)!=set(b):raise ValueError('Ambiguous/mismatched truth')
    return [(a[k],b[k]) for k in a]

def main():
    p=freeze();oldp=prior.read(PREVIOUS/'protocol.json');prior.verify(oldp)
    training=prior.read(TRAIN/'protocol.json');prior.verify(training)
    training_rows={r['member_id']:r for r in training['pool_rows']};aliases={};alias_paths=[]
    for m in p['members']:
        if m['condition']!='original_control':continue
        mid=m['source_id']+'-original';row=training_rows[mid]
        with Image.open(m['image_path']) as a,Image.open(row['image_path']) as b:
            if a.size!=b.size or a.convert('RGB').tobytes()!=b.convert('RGB').tobytes():raise ValueError('Original alias RGB mismatch')
        truth_key=lambda t:(t['class_name'],tuple(t['bbox_xyxy']))
        if Counter(map(truth_key,m['truth']))!=Counter(map(truth_key,row['full_truth']['objects'])):raise ValueError('Original alias complete truth mismatch')
        aliases[m['member_id']]=mid;alias_paths.append(Path(row['image_path']))
    trajectory=TRAIN.parent/'material-learning-trajectory-v1/analysis-v1/summary-v2.json'
    ts=prior.read(trajectory);prior.verify(ts)
    cohort={(t['cell'],t['member_id'],identity(t['truth'])) for t in ts['trajectories']
        if t['mode']=='ema' and t['planned_target'] and t['exposure_steps'] and t['classification']=='previously_hit_endpoint_miss'}
    if len(cohort)!=13:raise ValueError('Historical cohort drift')
    paths=[OUT/'protocol.json',PREVIOUS/'protocol.json',trajectory,TRAIN/'protocol.json',Path(__file__).resolve(),*alias_paths];details=[];counts=[]
    for seed in (7,17,27):
        nk,ok=f'D-{seed}',f'T-{seed}'
        np,op=OUT/(nk+'.json'),PREVIOUS/(ok+'.json')
        new,old=prior.read(np),prior.read(op);validate(new,nk,p);validate_old(old,ok,oldp);paths.extend((np,op))
        # Exposure sets intentionally change; classify rather than assume equal fit populations.
        olds={r['member_id']:r for r in old['rows']};news={r['member_id']:r for r in new['rows']}
        for m in p['members']:
            mid=m['member_id'];actual_mid=aliases.get(mid,mid);a,b=olds[mid],news[mid]
            if a['image_sha256']!=b['image_sha256']:raise ValueError('Different image')
            ah={x['truth_index'] for x in a['matches']};bh={x['truth_index'] for x in b['matches']}
            for i,j in match_truth(a['truth'],b['truth']):
                t=b['truth'][j];before=i in ah;after=j in bh
                details.append(dict(seed=seed,member_id=mid,source_id=m['source_id'],condition=m['condition'],truth=t,
                    image_sha256=m['image_sha256'],planned_target=t['object_id']==m['target'],
                    actual_member_id=actual_mid,exposures=p['actual_exposures'][nk].get(actual_mid,0),before_exposures=oldp['actual_exposures'][ok].get(actual_mid,0),before_hit=before,after_hit=after,
                    state='retained_hit' if before and after else 'loss' if before else 'gain' if after else 'retained_miss',
                    historical_loss_cohort=(ok,mid,identity(t)) in cohort,
                    before_miss=next((x for x in a['misses'] if x['truth_index']==i),None),
                    after_miss=next((x for x in b['misses'] if x['truth_index']==j),None)))
        for condition in sorted({m['condition'] for m in p['members']}):
            group=[x for x in details if x['seed']==seed and x['condition']==condition]
            target=[x for x in group if x['planned_target']];exposed=[x for x in target if x['exposures']]
            counts.append(dict(seed=seed,condition=condition,target_count=len(target),
                T_target_hits=sum(x['before_hit'] for x in target),D_target_hits=sum(x['after_hit'] for x in target),
                full_truth_count=len(group),T_full_hits=sum(x['before_hit'] for x in group),D_full_hits=sum(x['after_hit'] for x in group),
                actual_exposed_target_count=len(exposed),T_exposed_hits=sum(x['before_hit'] for x in exposed),D_exposed_hits=sum(x['after_hit'] for x in exposed)))
    exposed=[x for x in details if x['planned_target'] and x['exposures']]
    cohort_rows=[x for x in details if x['historical_loss_cohort']]
    if len(cohort_rows)!=13:raise ValueError('Incomplete matched cohort')
    groups={}
    for name in ('common_exposed','newly_exposed','original_now_zero','neither_exact_member_exposed'):
        subset=[x for x in details if x['planned_target'] and
            (('common_exposed' if x['exposures'] and x['before_exposures'] else
              'newly_exposed' if x['exposures'] else 'original_now_zero' if x['before_exposures'] else
              'neither_exact_member_exposed')==name)]
        groups[name]=dict(events=len(subset),before_hits=sum(x['before_hit'] for x in subset),
            after_hits=sum(x['after_hit'] for x in subset),transitions=dict(Counter(x['state'] for x in subset)))
    result=dict(original_pixel_verified_aliases=aliases,exposure_groups=groups,status='matched_member_fit_diagnosis_complete',counts=counts,details=details,
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
