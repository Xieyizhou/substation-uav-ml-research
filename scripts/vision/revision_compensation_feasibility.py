"""Count-only feasibility from hash-valid full-label reviews; never trains."""
import sys
from collections import Counter
from pathlib import Path
from scripts.vision.audit_revision_exposure import OUT as AUDIT,DESIGN,TRAIN,ROOT,read,verify,frozen,file_sha256
from scripts.vision.lineage_capped_control import OUT as POLICY,CONTROL,SECOND
from scripts.vision.structure_fit import truth_for

OUT=DESIGN/'compensation-feasibility-v1'


def reviewed_members(pool):
    idx={r['member_id']:r for r in pool};accepted=set();bad=set();paths=[]
    for root in (CONTROL,SECOND):
        ep,rp=root/'evidence.json',root/'review.json';e,r=read(ep),read(rp)
        verify(e);verify(r);paths += [ep,rp]
        ds={d['event_id']:d for d in r['decisions']}
        if len(ds)!=len(r['decisions']):raise ValueError('Duplicate review')
        if set(ds)!={l['event_id'] for f in e['events'] for l in f['labels']}:raise ValueError('Incomplete review')
        for f in e['events']:
            m=f['member']['member_id']
            if m not in idx:continue
            member=idx[m]
            for k in ('image','label'):
                if member[k+'_sha256']!=f['member'][k+'_sha256'] or file_sha256(Path(member[k+'_path']))!=member[k+'_sha256']:raise ValueError('Stale member')
                paths.append(Path(member[k+'_path']))
            truths=truth_for(member)
            if len(truths)!=len(f['labels']):raise ValueError('Full-label count mismatch')
            usable=True
            for t,l in zip(truths,f['labels']):
                d=ds[l['event_id']];crop=Path(l['crop_path'])
                if t['class_name']!=l['truth']['class_name'] or max(abs(a-b) for a,b in zip(t['bbox_xyxy'],l['truth']['bbox_xyxy']))>1e-4:raise ValueError('Full-label mismatch')
                if d['member_id']!=m or d['crop_sha256']!=l['crop_sha256'] or file_sha256(crop)!=l['crop_sha256']:raise ValueError('Stale crop decision')
                if d['image_sha256']!=member['image_sha256'] or d['label_sha256']!=member['label_sha256'] or not d['reason']:raise ValueError('Stale decision')
                paths.append(crop)
                usable &= d['status']=='identifiable_geometry_with_recorded_limits'
            (accepted if usable else bad).add(m)
    return accepted-bad,paths


def solve(rows,old,held,allow,groups):
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    n=len(rows);matrix=[];targets=[]
    for field,keys in [('subset',sorted({r['subset'] for r in rows})),('class',sorted({k for r in rows for k in r['class_instances']}))]:
        for k in keys:
            a=[int(r['subset']==k) if field=='subset' else r['class_instances'].get(k,0) for r in rows]
            matrix.append(a);targets.append(sum(v*old[r['member_id']] for v,r in zip(a,rows)))
    lo=[];hi=[]
    for r in rows:
        m=r['member_id'];v=old[m]
        if m in held:lo.append(0);hi.append(0)
        elif r['subset']=='hard_negative':lo.append(v);hi.append(v)
        else:lo.append(min(1,v));hi.append(2700 if m in allow and v>0 else v)
    lows=list(targets);highs=list(targets)
    for members in groups.values():
        a=[int(r['member_id'] in members) for r in rows];matrix.append(a);lows.append(0);highs.append(sum(old[m] for m in members))
    res=milp(np.zeros(n),integrality=np.ones(n),bounds=Bounds(lo,hi),constraints=LinearConstraint(np.array(matrix),lows,highs),options={'time_limit':30})
    if not res.success:return dict(status='infeasible' if res.status==2 else 'inconclusive',message=res.message,independent_infeasibility_certificate=False)
    x=np.rint(res.x).astype(int)
    if max(abs(res.x-x))>1e-6 or any(x<lo) or any(x>hi):raise ValueError('Invalid integer witness')
    values=np.array(matrix)@x
    if any(values<np.array(lows)) or any(values>np.array(highs)):raise ValueError('Constraint mismatch')
    return dict(status='integer_count_witness_verified',counts={r['member_id']:int(v) for r,v in zip(rows,x)},
        changes=[dict(member_id=r['member_id'],before=old[r['member_id']],after=int(v)) for r,v in zip(rows,x) if v!=old[r['member_id']]],
        witness_not_optimized=True)


def main():
    dest=OUT/'counts.json'
    if dest.exists():verify(read(dest));print('VALID_COUNTS_REUSED');return
    pp=TRAIN/'protocol.json';policy_path=POLICY/'protocol.json';ap=AUDIT/'audit.json'
    paths=[pp,policy_path,ap,Path(__file__)]
    for p in paths[:3]:verify(read(p))
    p,policy,a=read(pp),read(policy_path),read(ap);rows=p['pool_rows']
    reviewed,extra=reviewed_members(rows);paths+=extra
    newheld={r['member_id'] for r in a['units'][0]['members']};held=newheld|set(policy['held_member_ids'])
    allow=(reviewed & set(policy['increase_allowlist']))-held
    solver_file=next(Path(k) for k in policy['inputs'] if k.endswith('scipy/__init__.py'))
    if file_sha256(solver_file)!=policy['inputs'][str(solver_file)]:raise ValueError('Solver identity changed')
    sys.path.insert(0,str(solver_file.parent.parent));paths.append(solver_file)
    results={}
    for seed in (7,17,27):
        u=next(u for u in a['units'] if u['family']=='brightness_lr0005' and u['seed']==seed)
        ep=Path(u['exposure_receipt']);verify(read(ep));paths.append(ep);old=Counter(read(ep)['draws'])
        results[str(seed)]=solve(rows,old,held,allow,policy['risk_lineage_groups'])
        print(seed,results[str(seed)]['status'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    frozen(dest,dict(status='count_feasible_quality_preflight_not_ready' if all(r['status']=='integer_count_witness_verified' for r in results.values()) else 'compensation_not_established',
        results=results,reviewed_increase_candidates=sorted(allow),held_members=sorted(held),
        constraints=['Same subset and full-class instance totals','Previously zero members remain zero','Old and new held members zero',
            'Unknown/non-allowlisted members no increase','Risk lineage totals no increase','Negative counts fixed; no sequence generated, positions not changed'],
        quality_scope='Existing full-label content reviews only; not new hidden-instance completeness certification.',
        schedule_generated=False,training_ready=False,training_started=False,data_changed=False,
        inputs={str(x):file_sha256(x) for x in paths}))


if __name__=='__main__':main()
