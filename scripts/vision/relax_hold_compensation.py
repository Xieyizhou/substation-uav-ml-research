"""Broader positive-count feasibility only. No sequence export or training."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from scripts.vision.train_whole_image_hold import OUT as RUN,ROOT,ready,read,verify,frozen,file_sha256

OUT=RUN/'positive-redistribution-feasibility-v1'
NAMES=('transformer','switchgear','capacitor_bank','reactor')
SUBSETS=('base','regular','bridge_positive')

def tally(rows,counts):
    return dict(subsets={s:sum(counts.get(r['member_id'],0) for r in rows if r['subset']==s) for s in (*SUBSETS,'hard_negative')},
        classes={k:sum(counts.get(r['member_id'],0)*r['class_instances'].get(k,0) for r in rows) for k in NAMES})

def validate_counts(rows,old,new,held):
    idx={r['member_id']:r for r in rows}
    if len(idx)!=len(rows) or set(new)!=set(idx):raise ValueError('Missing/ambiguous membership')
    if any(type(v)!=int or v<0 for v in new.values()):raise ValueError('Noninteger or negative count')
    if any(new[m] for m in held):raise ValueError('Held image leaked')
    for m,r in idx.items():
        if r['subset']=='hard_negative' and new[m]!=old.get(m,0):raise ValueError('Negative exposure changed')
        if r['subset']!='hard_negative' and m not in held and new[m]<1:raise ValueError('Retained positive silently dropped')
    if sum(new.values())!=2700 or tally(rows,old)!=tally(rows,new):raise ValueError('Full class or subset exposure changed')

def solve(rows,old,held):
    import numpy as np
    from scipy.optimize import milp,LinearConstraint,Bounds
    positive=sorted([r for r in rows if r['subset'] in SUBSETS and r['member_id'] not in held],key=lambda r:r['member_id'])
    n=len(positive);size=2*n+1;z=n;matrix=[];lo=[];hi=[]
    target=tally(rows,old)
    for kind,keys in [('subsets',SUBSETS),('classes',NAMES)]:
        for key in keys:
            a=np.zeros(size)
            a[:n]=[int(r['subset']==key) if kind=='subsets' else r['class_instances'].get(key,0) for r in positive]
            matrix.append(a);lo.append(target[kind][key]);hi.append(target[kind][key])
    for i,r in enumerate(positive):
        original=old.get(r['member_id'],0)
        for coeff,bound in [({i:1,z:-1},original),({i:1,n+1+i:-1},original),({i:-1,n+1+i:-1},-original)]:
            a=np.zeros(size)
            for j,v in coeff.items():a[j]=v
            matrix.append(a);lo.append(-np.inf);hi.append(bound)
    lower=np.zeros(size);lower[:n]=1
    upper=np.full(size,2700.)
    constraint=LinearConstraint(np.array(matrix),np.array(lo),np.array(hi))
    c=np.zeros(size);c[z]=1
    first=milp(c,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':60,'mip_rel_gap':0.0})
    if not first.success:raise ValueError(f'Peak-increment optimum not established: {first.message}')
    cap=int(round(first.x[z]));upper[z]=cap;c[z]=0;c[n+1:]=1
    second=milp(c,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':60,'mip_rel_gap':0.0})
    if not second.success:raise ValueError(f'Minimum-change optimum not established: {second.message}')
    if max(abs(second.x-np.rint(second.x)))>1e-6:raise ValueError('Fractional solver output rejected')
    new={r['member_id']:(old.get(r['member_id'],0) if r['subset']=='hard_negative' else 0) for r in rows}
    for i,r in enumerate(positive):new[r['member_id']]=int(round(second.x[i]))
    validate_counts(rows,old,new,held)
    if max(new[r['member_id']]-old.get(r['member_id'],0) for r in positive)>cap:raise ValueError('Peak increment constraint violated')
    return new,dict(peak_extra_exposure=cap,retained_positive_L1_change=sum(abs(new[r['member_id']]-old.get(r['member_id'],0)) for r in positive),
        stage1_status=int(first.status),stage2_status=int(second.status),stage1_gap=float(first.mip_gap),stage2_gap=float(second.mip_gap))

def build():
    import scipy,numpy
    p=ready();prior=RUN/'instance-exposure-compensation-feasibility-v1/analysis.json';verify(read(prior))
    rows=p['pool_rows'];held=set(p['held_member_ids']);results={};idx={r['member_id']:r for r in rows}
    for seed in (7,17,27):
        old=Counter(p['schedules'][f'reference-450-{seed}']);new,obj=solve(rows,old,held)
        changes=[dict(member_id=m,subset=idx[m]['subset'],lineage_id=idx[m]['lineage_id'],before=old[m],after=new[m],delta=new[m]-old[m],class_instances=idx[m]['class_instances']) for m in sorted(idx) if new[m]!=old[m]]
        group_before=Counter();group_after=Counter()
        for r in rows:group_before[r['lineage_id']]+=old[r['member_id']];group_after[r['lineage_id']]+=new[r['member_id']]
        results[str(seed)]=dict(counts=new,objective=obj,changes=changes,totals=tally(rows,new),minimum_changed_slots=sum(max(0,new[m]-old[m]) for m in new),
            maximum_member_exposure=max(new.values()),registered_lineage_before=dict(group_before),registered_lineage_after=dict(group_after),
            active_members=sum(v>0 for v in new.values()),negative_slot_policy='Keep every original negative position/member unchanged; no sequence constructed yet.')
        print('SOLVED',seed,obj,'CHANGED_SLOTS',results[str(seed)]['minimum_changed_slots'],flush=True)
    OUT.mkdir(exist_ok=True)
    paths=[RUN/'protocol.json',RUN/'ready.json',prior,Path(__file__),Path(scipy.__file__)]
    return frozen(OUT/'counts.json',dict(status='integer_count_feasible_sequence_and_quality_checks_pending',results=results,
        constraints=['2700 total','same positive subset quotas','same four full-label class totals','every negative member count unchanged','held four zero','every retained positive at least once'],
        objective='Lexicographically minimize largest extra exposure of any retained positive, then total absolute count change. Not a validated quality threshold.',
        solver_versions=dict(scipy=scipy.__version__,numpy=numpy.__version__),
        tie_policy='A frozen integer count witness; solver ties may differ across builds, no sequence is implied.',
        new_training_sequence=False,training_started=False,all_quality_risks_resolved=False,
        inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--solver-path');a=ap.parse_args()
    if a.solver_path:sys.path.insert(0,a.solver_path);build()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
