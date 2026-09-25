"""Count-only feasibility with twelve reviewed images capped at reference exposure."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from scripts.vision.relax_hold_compensation import RUN, ROOT, ready, read, verify, frozen, file_sha256, tally, validate_counts, SUBSETS, NAMES
from scripts.vision.audit_redistribution_targets import OUT as ATTRIBUTION
from scripts.vision.replay_redistribution_occlusion import OUT as REPLAY

OUT = RUN / 'risk-capped-redistribution-feasibility-v1'

def validate_capped(rows, old, new, held, risks):
    validate_counts(rows, old, new, held)
    ids={r['member_id'] for r in rows}
    if not risks or not set(risks)<=ids or set(risks)&held: raise ValueError('Unresolved or held risk member')
    if any(new[m]>old.get(m,0) for m in risks): raise ValueError('Risk image exposure increased')

def solve(rows, old, held, risks):
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    positive=sorted((r for r in rows if r['subset'] in SUBSETS and r['member_id'] not in held),key=lambda r:r['member_id'])
    n=len(positive); size=2*n+1; z=n; matrix=[]; lo=[]; hi=[]; target=tally(rows,old)
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
    lower=np.zeros(size);lower[:n]=1;upper=np.full(size,2700.)
    for i,r in enumerate(positive):
        if r['member_id'] in risks:upper[i]=old.get(r['member_id'],0)
    c=np.zeros(size);c[z]=1;constraint=LinearConstraint(np.array(matrix),lo,hi)
    first=milp(c,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':60,'mip_rel_gap':0.0})
    if first.status==2:return None,dict(status='solver_reports_infeasible',message=first.message,independent_infeasibility_certificate=False)
    if not first.success:return None,dict(status='solver_inconclusive',message=first.message)
    cap=int(round(first.x[z]));upper[z]=cap;c[z]=0;c[n+1:]=1
    second=milp(c,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=constraint,options={'time_limit':60,'mip_rel_gap':0.0})
    if not second.success:return None,dict(status='second_stage_inconclusive',message=second.message,established_peak_extra=cap)
    if max(abs(second.x-np.rint(second.x)))>1e-6:raise ValueError('Fractional output')
    new={r['member_id']:old.get(r['member_id'],0) if r['subset']=='hard_negative' else 0 for r in rows}
    for i,r in enumerate(positive):new[r['member_id']]=int(round(second.x[i]))
    validate_capped(rows,old,new,held,risks)
    if max(new[m]-old.get(m,0) for m in new)!=cap:raise ValueError('Peak objective mismatch')
    return new,dict(status='optimal_integer_witness_verified',peak_extra_exposure=cap,
        retained_positive_L1_change=sum(abs(new[r['member_id']]-old.get(r['member_id'],0)) for r in positive),
        stage1_gap=float(first.mip_gap),stage2_gap=float(second.mip_gap))

def build():
    import scipy,numpy
    if (OUT/'counts.json').exists():verify(read(OUT/'counts.json'));print('EXISTING_COUNTS_REVALIDATED');return
    p=ready();review=read(ATTRIBUTION/'review.json');verify(review)
    for path in (REPLAY/'review.json',REPLAY/'completion.json'):verify(read(path))
    risks={d['member_id']:d['event_id'] for d in review['decisions']}
    if len(risks)!=12:raise ValueError('Risk scope incomplete')
    held=set(p['held_member_ids']); rows=p['pool_rows']; idx={r['member_id']:r for r in rows}
    if not set(risks)<=set(idx) or set(risks)&held:raise ValueError('Risk membership mismatch')
    results={}
    for seed in (7,17,27):
        old=Counter(p['schedules'][f'reference-450-{seed}']);new,obj=solve(rows,old,held,risks)
        x=dict(objective=obj)
        if new is not None:
            changes=[dict(member_id=m,subset=idx[m]['subset'],before=old[m],after=new[m],delta=new[m]-old[m]) for m in sorted(new) if new[m]!=old[m]]
            x.update(counts=new,totals=tally(rows,new),changes=changes,minimum_changed_slots=sum(max(0,new[m]-old[m]) for m in new),
                risk_exposures=[dict(event_id=e,member_id=m,before=old[m],after=new[m]) for m,e in sorted(risks.items())],
                maximum_member_exposure=max(new.values()),active_members=sum(v>0 for v in new.values()))
        results[str(seed)]=x;print('SEED',seed,obj,'changed_slots',x.get('minimum_changed_slots'),flush=True)
    OUT.mkdir(exist_ok=True)
    paths=[RUN/'protocol.json',RUN/'ready.json',ATTRIBUTION/'review.json',REPLAY/'review.json',REPLAY/'completion.json',Path(__file__),ROOT/'scripts/vision/relax_hold_compensation.py',Path(scipy.__file__)]
    frozen(OUT/'counts.json',dict(status='count_feasible_not_training_ready' if all('counts' in x for x in results.values()) else 'feasibility_not_established',
        results=results,risk_members=risks,cap_reference='Each seed original reference-450 schedule; not previous proposed redistribution counts.',
        constraints=['2700 total','same subset totals','same full four-class instance totals','each negative count unchanged','four held images zero','each retained positive at least once','twelve risk images no increase'],
        objective='Minimize maximum positive increment, then retained-positive L1 under that optimum; not a quality admission rule.',
        solver_versions=dict(scipy=scipy.__version__,numpy=numpy.__version__),
        schedule_generated=False,training_started=False,quality_gate_passed=False,
        inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--solver-path');a=ap.parse_args()
    if a.solver_path:sys.path.insert(0,a.solver_path);build()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
