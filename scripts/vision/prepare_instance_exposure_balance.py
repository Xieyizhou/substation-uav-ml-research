"""Bounded integer optimization of training-label exposure, without eval-driven selection."""
import math
import random
from collections import Counter
from pathlib import Path
from scripts.vision.run_visibility_quality_training import OUT as PRIOR, read, save, file_sha256, verify_tree
from scripts.vision.analyze_visibility_quality_results import OUT as DIAG
from scripts.vision.exposure_protocol import ROOT, SEEDS, NAMES, exposures
from scripts.vision.canonical_batch_order import canonicalize

OUT=PRIOR/'full-instance-exposure-balance-v1'
QUOTAS=dict(base=216,regular=156,bridge_positive=120)

def sort_positive_slots(draws,lookup):
    result=list(draws)
    if len(result)%6:raise ValueError('Incomplete batch')
    for start in range(0,len(result),6):
        slots=[i for i in range(start,start+6) if lookup[result[i]]['subset']!='hard_negative']
        for index,mid in zip(slots,sorted(result[i] for i in slots)):result[index]=mid
    return result

def optimize(rows, old, seed):
    import numpy as np
    import scipy
    from scipy.optimize import milp, Bounds, LinearConstraint
    positive=sorted((r for r in rows if r['subset'] in QUOTAS),key=lambda r:r['member_id'])
    n=len(positive); size=2*n+2; counts=Counter(old)
    upper=[2*math.ceil(QUOTAS[r['subset']]/sum(x['subset']==r['subset'] for x in positive)) for r in positive]
    matrix=[];lo=[];hi=[]
    def constraint(v,l,h): matrix.append(v);lo.append(l);hi.append(h)
    for subset,q in QUOTAS.items():
        v=np.zeros(size);v[:n]=[r['subset']==subset for r in positive];constraint(v,q,q)
    for name in NAMES:
        v=np.zeros(size);v[:n]=[r['class_instances'].get(name,0) for r in positive]
        v[-2]=-1;constraint(v,-np.inf,0)  # class <= maximum
        v=v.copy();v[-2]=0;v[-1]=-1;constraint(v,0,np.inf)  # class >= minimum
    for i,r in enumerate(positive):
        v=np.zeros(size);v[i]=1;v[n+i]=-1;constraint(v,-np.inf,counts[r['member_id']])
        v=np.zeros(size);v[i]=-1;v[n+i]=-1;constraint(v,-np.inf,-counts[r['member_id']])
    bounds=Bounds([1]*n+[0]*(n+2),upper+[np.inf]*(n+2))
    objectives=[]
    def solve(cost):
        result=milp(cost,integrality=np.ones(size),bounds=bounds,constraints=LinearConstraint(np.array(matrix),lo,hi),options={'time_limit':60,'mip_rel_gap':0})
        if not result.success: raise ValueError(f'No proven optimum: {result.message}')
        rounded=np.rint(result.x).astype(int)
        if np.max(np.abs(result.x-rounded))>1e-5: raise ValueError('Noninteger solution')
        return rounded,result
    cost=np.zeros(size);cost[-2]=1;cost[-1]=-1
    x,result=solve(cost);spread=int(cost@x);constraint(cost.copy(),spread,spread);objectives.append(spread)
    cost=np.zeros(size);cost[n:2*n]=1
    x,result=solve(cost);change=int(cost@x);constraint(cost.copy(),change,change);objectives.append(change)
    order=list(range(n));random.Random(f'instance-balance-v1:{seed}').shuffle(order)
    cost=np.zeros(size)
    for rank,i in enumerate(order):cost[i]=rank+1
    x,result=solve(cost)
    chosen={r['member_id']:int(x[i]) for i,r in enumerate(positive)}
    # Independent integer feasibility and objective calculation.
    for subset,q in QUOTAS.items():
        if sum(chosen[r['member_id']] for r in positive if r['subset']==subset)!=q: raise ValueError('Quota failure')
    if any(not 1<=chosen[r['member_id']]<=upper[i] for i,r in enumerate(positive)): raise ValueError('Member cap failure')
    classes=Counter()
    for r in positive:
        for name,value in r['class_instances'].items():classes[name]+=value*chosen[r['member_id']]
    if max(classes.get(k,0) for k in NAMES)-min(classes.get(k,0) for k in NAMES)!=spread: raise ValueError('Range objective failure')
    if sum(abs(chosen[r['member_id']]-counts[r['member_id']]) for r in positive)!=change: raise ValueError('L1 objective failure')
    lookup={r['member_id']:r for r in rows};remaining=chosen.copy();draws=[];holes=[]
    for index,mid in enumerate(old):
        if lookup[mid]['subset']=='hard_negative':draws.append(mid)
        elif remaining[mid]>0:draws.append(mid);remaining[mid]-=1
        else:draws.append(None);holes.append(index)
    rng=random.Random(f'instance-balance-fill-v1:{seed}')
    for subset in QUOTAS:
        extras=[mid for mid,num in sorted(remaining.items()) if lookup[mid]['subset']==subset for _ in range(num)];rng.shuffle(extras)
        positions=[i for i in holes if lookup[old[i]]['subset']==subset]
        if len(extras)!=len(positions):raise ValueError('Replacement mismatch')
        for index,mid in zip(positions,extras):draws[index]=mid
    draws=sort_positive_slots(draws,lookup)
    for i,mid in enumerate(old):
        if lookup[mid]['subset']=='hard_negative' and draws[i]!=mid:raise ValueError('Negative position changed')
    if any(Counter(draws)[mid]!=count for mid,count in chosen.items()):raise ValueError('Effective count mismatch')
    return draws,dict(class_exposure=dict(classes),optimal_range=spread,optimal_L1_change=change,solver='scipy.optimize.milp',scipy_version=scipy.__version__,tie_objective=int(cost@x),proof='MILP success, zero requested gap; independent integer quota, cap, range and L1 checks')

def main():
    target=OUT/'protocol.json'
    if target.exists():verify_tree(target);print('VERIFIED_EXISTING');return
    verify_tree(DIAG/'completion.json');p=read(PRIOR/'protocol.json')
    OUT.mkdir(parents=True,exist_ok=True);schedules={};audit={};datasets={}
    inputs={str(path):file_sha256(path) for path in (DIAG/'completion.json',PRIOR/'protocol.json',Path(__file__),ROOT/'scripts/vision/run_instance_exposure_balance.py',ROOT/'tests/test_instance_exposure_balance.py')}
    for seed in SEEDS:
        old=p['schedules'][f'Q-100-{seed}'];draws,a=optimize(p['pool_rows'],old,seed)
        before=p['exposures'][f'Q-100-{seed}']['class_instance_exposure'];a['prior_exposure']=before
        if a['optimal_range']>=max(before.values())-min(before.values()):raise ValueError('No exposure-range improvement; stop sampling path')
        audit[str(seed)]=a
        for steps in (100,300):
            key=f'I-{steps}-{seed}';schedules[key]=draws*(steps//100)
            lookup={r['member_id']:r for r in p['pool_rows']}
            listing=OUT/f'{key}.txt';listing.write_text('\n'.join(lookup[x]['image_path'] for x in sorted(set(draws)))+'\n')
            dataset=OUT/f'{key}.yaml';dataset.write_text(f'path: {OUT}\ntrain: {listing}\nval: {listing}\nnames: [transformer, switchgear, capacitor_bank, reactor]\n');datasets[key]=str(dataset)
            for path in (listing,dataset):inputs[str(path)]=file_sha256(path)
        print('OPTIMAL',seed,a,flush=True)
    save(target,dict(status='frozen_before_training',pool_rows=p['pool_rows'],schedules=schedules,datasets=datasets,exposures={k:exposures(p['pool_rows'],v) for k,v in schedules.items()},optimization=audit,controls=p['controls'],acceptance_policy=p['acceptance_policy'],retention=p['retention'],candidate_priority=['I-100','I-300'],max_attempts=3,inputs=inputs,interpretation='Train-label-only bounded class-exposure range minimization; full-image cooccurrences retained. Not independent scene expansion or pure class-isolated intervention. Negative members and exposure positions unchanged; only positive slots sorted within each batch. Replacement also changes positive batch composition and reduction order; not a pure class-count causal effect.',preflight_correction='Initial full-batch sorting rejected before training because negative positions changed. Corrected to positive-slot-only deterministic sorting and added adversarial test.'))
    print('FROZEN',OUT,flush=True)

if __name__=='__main__':main()
