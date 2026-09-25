"""Lexicographic count optimization; no dataset or sampling sequence generated."""
import sys
from collections import Counter
from pathlib import Path
from scripts.vision import revision_compensation_feasibility as prior

OUT=prior.OUT/'minimax-v1'


def optimize(rows,old,held,allow,groups):
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    n=len(rows);size=2*n+1;z=n;matrix=[];low=[];high=[]
    for field,keys in [('subset',sorted({r['subset'] for r in rows})),('class',sorted({k for r in rows for k in r['class_instances']}))]:
        for k in keys:
            a=np.zeros(size);a[:n]=[int(r['subset']==k) if field=='subset' else r['class_instances'].get(k,0) for r in rows]
            target=sum(v*old[r['member_id']] for v,r in zip(a,rows));matrix.append(a);low.append(target);high.append(target)
    for members in groups.values():
        a=np.zeros(size);a[:n]=[int(r['member_id'] in members) for r in rows]
        matrix.append(a);low.append(0);high.append(sum(old[m] for m in members))
    lower=np.zeros(size);upper=np.full(size,5400.)
    for i,r in enumerate(rows):
        m=r['member_id'];v=old[m]
        if m in held:upper[i]=0
        elif r['subset']=='hard_negative':lower[i]=upper[i]=v
        else:lower[i]=min(1,v);upper[i]=2700 if m in allow and v>0 else v
        for coeff,bound in [({i:1,z:-1},v),({i:1,n+1+i:-1},v),({i:-1,n+1+i:-1},-v)]:
            a=np.zeros(size)
            for j,value in coeff.items():a[j]=value
            matrix.append(a);low.append(-np.inf);high.append(bound)
    constraint=LinearConstraint(np.array(matrix),low,high);c=np.zeros(size);c[z]=1
    first=milp(c,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':30,'mip_rel_gap':0})
    if not first.success:return dict(status='stage1_not_established',message=first.message)
    peak=int(round(first.x[z]));upper[z]=peak;c[z]=0;c[n+1:]=1
    second=milp(c,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':30,'mip_rel_gap':0})
    if not second.success:return dict(status='stage2_not_established',peak=peak,message=second.message)
    x=np.rint(second.x).astype(int);values=np.array(matrix)@x
    if max(abs(second.x-x))>1e-6 or any(x<lower) or any(x>upper) or any(values<np.array(low)) or any(values>np.array(high)):raise ValueError('Invalid integer witness')
    counts={r['member_id']:int(v) for r,v in zip(rows,x)}
    changes=[dict(member_id=m,before=old[m],after=v,delta=v-old[m]) for m,v in counts.items() if v!=old[m]]
    if max((v-old[m] for m,v in counts.items()),default=0)!=peak:raise ValueError('Peak mismatch')
    l1=sum(abs(v-old[m]) for m,v in counts.items())
    if l1!=int(round(second.fun)):raise ValueError('L1 objective mismatch')
    return dict(status='lexicographic_integer_optimum_verified',peak_extra=peak,total_L1=l1,
        reallocated_slots=l1//2,max_member_count=max(counts.values()),counts=counts,changes=changes,
        solver_gaps=[float(first.mip_gap),float(second.mip_gap)],
        tied_optima_possible=True,sequence_generated=False)


def main():
    dest=OUT/'counts.json'
    if dest.exists():prior.verify(prior.read(dest));print('VALID_OPTIMUM_REUSED');return
    fp=prior.OUT/'counts.json';pp=prior.TRAIN/'protocol.json';ap=prior.AUDIT/'audit.json';rp=prior.POLICY/'protocol.json'
    paths=[fp,pp,ap,rp,Path(__file__)]
    for p in paths[:4]:prior.verify(prior.read(p))
    f,p,a,policy=map(prior.read,(fp,pp,ap,rp));rows=p['pool_rows'];idx={r['member_id']:r for r in rows}
    reviewed,extra=prior.reviewed_members(rows);paths+=extra
    allow=set(f['reviewed_increase_candidates'])
    if not allow<=reviewed:raise ValueError('Review allowlist changed')
    sf=next(Path(k) for k in policy['inputs'] if k.endswith('scipy/__init__.py'))
    if prior.file_sha256(sf)!=policy['inputs'][str(sf)]:raise ValueError('Changed solver')
    sys.path.insert(0,str(sf.parent.parent));paths.append(sf)
    results={};queue={}
    for seed in (7,17,27):
        unit=next(u for u in a['units'] if u['family']=='brightness_lr0005' and u['seed']==seed)
        ep=Path(unit['exposure_receipt']);prior.verify(prior.read(ep));paths.append(ep)
        result=optimize(rows,Counter(prior.read(ep)['draws']),set(f['held_members']),allow,policy['risk_lineage_groups']);results[str(seed)]=result
        for change in result.get('changes',[]):
            if change['delta']>0:
                m=change['member_id'];queue.setdefault(m,dict(member=idx[m],seed_increments={},status='hidden_instance_completeness_pending'))['seed_increments'][str(seed)]=change['delta']
        print(seed,{k:v for k,v in result.items() if k not in ('counts','changes')},flush=True)
    for r in rows:
        for k in ('image','label'):
            path=Path(r[k+'_path'])
            if prior.file_sha256(path)!=r[k+'_sha256']:raise ValueError('Pool input changed')
            paths.append(path)
    OUT.mkdir(parents=True,exist_ok=True)
    prior.frozen(dest,dict(status='optimized_counts_quality_pending' if all(r['status']=='lexicographic_integer_optimum_verified' for r in results.values()) else 'optimization_incomplete',
        results=results,review_queue=list(queue.values()),held_members=f['held_members'],
        objective='Minimize maximum member increase, then total absolute count change; no model-score selection.',
        training_ready=False,training_started=False,schedule_generated=False,data_changed=False,
        limitation='Existing registered-lineage caps retained; no new independence or missing-instance certification. Windows require a later frozen sequence.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('REVIEW_QUEUE',len(queue))


if __name__=='__main__':main()
