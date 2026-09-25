"""Two-stage integer exposure compensation; never trains or changes labels."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from scripts.vision.diagnose_small_scale_order_fit import OUT,TRAIN,prior

SOLVER=prior.ROOT/'data/research/ml_training_recovery_v1/tools/exposure-solver-v1'
DEST=OUT/'reviewed-dataset-v1/exposure-control-v1'


def verify_counts(rows,old,new,allowed):
    ids={r['member_id'] for r in rows}
    if set(new)!=ids or any(type(v)!=int or v<0 for v in new.values()):raise ValueError('Invalid integer member counts')
    for r in rows:
        mid=r['member_id'];before=old.get(mid,0);after=new[mid]
        if mid not in allowed and after:raise ValueError('Held member restored')
        if before==0 and after:raise ValueError('Zero-exposure member restored')
        if not r['class_instances'] and after!=before:raise ValueError('Negative count changed')
        if mid in allowed and before>0 and after<1:raise ValueError('Previously exposed approved member dropped')
    for subset in {r['subset'] for r in rows}:
        if sum(new[r['member_id']] for r in rows if r['subset']==subset)!=sum(old.get(r['member_id'],0) for r in rows if r['subset']==subset):raise ValueError('Subset budget drift')
    for name in {k for r in rows for k in r['class_instances']}:
        if sum(new[r['member_id']]*r['class_instances'].get(name,0) for r in rows)!=sum(old.get(r['member_id'],0)*r['class_instances'].get(name,0) for r in rows):raise ValueError('Full-class exposure drift')


def solve(rows,old,allowed):
    from scipy.optimize import milp,Bounds,LinearConstraint
    n=len(rows);size=2*n+1;z=n;t=n+1
    lower=np.zeros(size);upper=np.full(size,np.inf);integrality=np.zeros(size)
    integrality[:n+1]=1;matrix=[];lo=[];hi=[]
    def add(a,l,u):matrix.append(a);lo.append(l);hi.append(u)
    for i,r in enumerate(rows):
        mid=r['member_id'];before=old.get(mid,0)
        if mid not in allowed or before==0:upper[i]=0
        elif not r['class_instances']:lower[i]=upper[i]=before
        else:lower[i]=1;upper[i]=6600
        a=np.zeros(size);a[i]=1;a[z]=-1;add(a,-np.inf,before)
        a=np.zeros(size);a[i]=1;a[t+i]=-1;add(a,-np.inf,before)
        a=np.zeros(size);a[i]=-1;a[t+i]=-1;add(a,-np.inf,-before)
    for field,keys in [('subset',sorted({r['subset'] for r in rows})),('class',sorted({k for r in rows for k in r['class_instances']}))]:
        for key in keys:
            a=np.zeros(size)
            for i,r in enumerate(rows):a[i]=int(r['subset']==key) if field=='subset' else r['class_instances'].get(key,0)
            target=sum(a[i]*old.get(r['member_id'],0) for i,r in enumerate(rows));add(a,target,target)
    constraint=LinearConstraint(np.asarray(matrix),lo,hi)
    c=np.zeros(size);c[z]=1
    first=milp(c,integrality=integrality,bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':30,'mip_rel_gap':0})
    if not first.success:return dict(status='infeasible' if first.status==2 else 'inconclusive',message=first.message,stage=1)
    best=int(round(first.x[z]));lower[z]=upper[z]=best;c[:]=0;c[t:]=1
    second=milp(c,integrality=integrality,bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':30,'mip_rel_gap':0})
    if not second.success:return dict(status='inconclusive',message=second.message,stage=2,optimal_max_increase=best)
    if np.max(np.abs(second.x[:n]-np.rint(second.x[:n])))>1e-6:raise ValueError('Nonintegral solution')
    counts={r['member_id']:int(round(second.x[i])) for i,r in enumerate(rows)}
    verify_counts(rows,old,counts,allowed)
    maximum=max(counts[r['member_id']]-old.get(r['member_id'],0) for r in rows)
    l1=sum(abs(counts[r['member_id']]-old.get(r['member_id'],0)) for r in rows)
    if maximum!=best or abs(l1-second.fun)>1e-5:raise ValueError('Objective witness mismatch')
    return dict(status='two_stage_optimal_integer_witness_verified',counts=counts,optimal_max_increase=best,
        optimal_total_absolute_count_change=l1,maximum_member_count=max(counts.values()),
        changed_members=[dict(member_id=r['member_id'],before=old.get(r['member_id'],0),after=counts[r['member_id']]) for r in rows if counts[r['member_id']]!=old.get(r['member_id'],0)],
        solver_optimality={'first_status':int(first.status),'second_status':int(second.status),'first_gap':float(first.mip_gap),'second_gap':float(second.mip_gap)})


def run():
    sys.path.insert(0,str(SOLVER));import scipy
    pp=OUT/'protocol.json';mp=OUT/'reviewed-dataset-v1/manifest.json';dp=TRAIN/'design.json'
    p,m,d=[prior.read(path) for path in (pp,mp,dp)]
    for r in (p,m,d):prior.verify(r)
    rows=p['members'];allowed={r['member_id'] for r in m['members']}
    allowed-=set(d['held_members'])
    paths=[pp,mp,dp,Path(__file__).resolve(),Path(scipy.__file__),Path(np.__file__)]
    paths+=list(Path(scipy.__file__).parent.rglob('*.so'))
    paths+=list(SOLVER.glob('scipy-*.dist-info/RECORD'))
    DEST.mkdir(exist_ok=True);results={}
    for key in sorted(d['schedules']):
        old=dict(Counter(d['schedules'][key]))
        if old!=p['actual_exposures'][key]:raise ValueError('Historical actual/planned exposure mismatch')
        dest=DEST/(key+'.json')
        if dest.exists():r=prior.read(dest);prior.verify(r)
        else:
            result=solve(rows,old,allowed)
            r=prior.frozen(dest,dict(key=key,**result,solver_version=scipy.__version__,numpy_version=np.__version__,
                constraints=['Original subset budgets and global full-class instance exposures exact.',
                    'All 53 quality-held and previously held members zero; previously zero members remain zero.',
                    'All empty-full-label negative member counts fixed; approved previously exposed members at least once.',
                    'Minimize maximum member increase, then total absolute member-count change.'],
                scope='Count-only feasibility; sequence, negative positions, brightness, per-50-step lineage exposure and training preflight still required.',
                training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))
        results[key]=r;print(key,r['status'],r.get('optimal_max_increase'),flush=True)
    return results


if __name__=='__main__':run()
