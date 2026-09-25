"""Bounded fixed-budget 54->162 new-member dose design; never trains."""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from scripts.vision.compensated_pool_fit import OUT as FIT, check as check_fit
from scripts.vision import train_compensated_material as source
from scripts.vision.build_compensated_error_review import DEST, prior
from scripts.vision.structure_fit import truth_for

OUT=DEST/'reviewed-dose-control-v1'
VERSION='reviewed-compensated-dose-v1'
KEYS=source.KEYS
CLASSES=('transformer','switchgear','capacitor_bank','reactor')


def rank(*parts):return hashlib.sha256(':'.join(map(str,(VERSION,*parts))).encode()).hexdigest()


def vector(row):return [1]+[row['class_instances'].get(c,0) for c in CLASSES]


def checks(old,p):
    rows={r['member_id']:r for r in p['pool_rows']}
    if len(rows)!=277 or p['pool_rows']!=old['pool_rows']:raise ValueError('Pool altered')
    for key in KEYS:
        before=old['schedules'][key];after=p['schedules'][key]
        a,b=Counter(before),Counter(after)
        if len(after)!=2700 or p['brightness_factors'][key]!=old['brightness_factors'][key]:raise ValueError('Budget or brightness changed')
        if p['training_config'][key]!=old['training_config'][key]:raise ValueError('Training hyperparameters changed')
        if Counter(rows[m]['subset'] for m in before)!=Counter(rows[m]['subset'] for m in after):raise ValueError('Subset budget changed')
        for cls in CLASSES:
            if sum(a[m]*r['class_instances'].get(cls,0) for m,r in rows.items())!=sum(b[m]*r['class_instances'].get(cls,0) for m,r in rows.items()):raise ValueError('Class exposure changed')
        for m,r in rows.items():
            if 'full_truth' in r:
                if b[m]!=3*a[m]:raise ValueError('New member dose not exactly threefold')
            elif not 0<=b[m]<=a[m]:raise ValueError('Old member increased')
            elif a[m]>0 and b[m]<1:raise ValueError('Existing exposed member lost')
        if set(after)&set(old['held_members']):raise ValueError('Held member restored')
        for i,(m,n) in enumerate(zip(before,after,strict=True)):
            if m!=n and (rows[m]['subset']!='bridge_positive' or 'full_truth' in rows[m] or 'full_truth' not in rows[n]):raise ValueError('Illegal replacement position')
        if p['initialization']!=old['initialization'] or p['evaluation']!=old['evaluation']:raise ValueError('Init/evaluation drift')
    for seed in (7,17,27):
        a,b=(p['schedules'][f'{arm}-{seed}'] for arm in ('V','VM'))
        for x,y in zip(a,b,strict=True):
            if x!=y and (rows[x].get('pair_id')!=rows[y].get('pair_id') or rows[x]['full_truth']!=rows[y]['full_truth']):raise ValueError('Unpaired complete labels')


def solve(old,seed,milp,Bounds,LinearConstraint):
    import numpy as np
    rows={r['member_id']:r for r in old['pool_rows']};counts=Counter(old['schedules'][f'V-{seed}'])
    selected=sorted((r for r in rows.values() if 'full_truth' not in r and r['subset']=='bridge_positive' and counts[r['member_id']]>1),key=lambda r:r['member_id'])
    extra={m:2*n for m,n in counts.items() if 'full_truth' in rows[m]}
    target=np.array([sum(n*vector(rows[m])[j] for m,n in extra.items()) for j in range(5)],dtype=float)
    n=len(selected);A=np.zeros((5+n,n+1));A[:5,:n]=np.array([vector(r) for r in selected]).T
    for i in range(n):A[5+i,i]=1;A[5+i,n]=-1
    lower=np.zeros(n+1);upper=np.array([counts[r['member_id']]-1 for r in selected]+[108],dtype=float)
    lo=np.r_[target,np.full(n,-np.inf)];hi=np.r_[target,np.zeros(n)]
    objective=np.r_[np.zeros(n),1.]
    first=milp(objective,integrality=np.ones(n+1),bounds=Bounds(lower,upper),constraints=LinearConstraint(A,lo,hi),options={'time_limit':30,'mip_rel_gap':0.0})
    if first.status!=0:return dict(status='count_not_established',solver_status=int(first.status),message=first.message)
    optimum=int(round(first.x[-1]));lower[-1]=upper[-1]=optimum
    # Stable, predeclared tie objective. Freeze the integer witness; never select by model scores.
    order=sorted(range(n),key=lambda i:rank(seed,selected[i]['member_id'],'removal'))
    c=np.zeros(n+1)
    for weight,i in enumerate(order,1):c[i]=weight
    second=milp(c,integrality=np.ones(n+1),bounds=Bounds(lower,upper),constraints=LinearConstraint(A,lo,hi),options={'time_limit':30,'mip_rel_gap':0.0})
    if second.status!=0:return dict(status='count_not_established',solver_status=int(second.status),message=second.message)
    x=np.rint(second.x).astype(int)
    if np.max(abs(second.x-x))>1e-6 or any(x<lower) or any(x>upper) or any(A@x<lo) or any(A@x>hi):raise ValueError('Invalid count witness')
    return dict(status='exact_threefold_dose_witness',seed=seed,removed={r['member_id']:int(v) for r,v in zip(selected,x[:-1]) if v},
        extra_original_members=extra,additional_positions=108,max_old_member_decrement=optimum,
        tie_objective=float(second.fun),no_score_based_selection=True)


def schedules(old,seed,witness):
    rows={r['member_id']:r for r in old['pool_rows']};v=list(old['schedules'][f'V-{seed}']);vm=list(old['schedules'][f'VM-{seed}'])
    pos=[]
    for mid,n in sorted(witness['removed'].items()):
        choices=sorted((i for i,x in enumerate(v) if x==mid),key=lambda i:rank(seed,mid,i))
        if len(choices)<n:raise ValueError('Removal overflow')
        pos+=choices[:n]
    extra=witness['extra_original_members'];items=[]
    for cycle in range(max(extra.values())):
        items+=sorted((m for m,n in extra.items() if n>cycle),key=lambda m:rank(seed,m,cycle))
    if len(items)!=len(pos) or len(pos)!=108:raise ValueError('Position mismatch')
    oldvm=Counter(vm);remaining={m:2*n for m,n in oldvm.items() if 'full_truth' in rows[m]};used=Counter()
    for i,mid in zip(sorted(pos),items,strict=True):
        v[i]=mid
        options=[m for m,n in remaining.items() if n>0 and rows[m]['pair_id']==rows[mid]['pair_id']]
        if not options:raise ValueError('Paired appearance allocation exhausted')
        choice=min(options,key=lambda m:(used[m],rank(seed,mid,m)))
        vm[i]=choice;used[choice]+=1;remaining[choice]-=1
    if any(remaining.values()):raise ValueError('Incomplete appearance allocation')
    return v,vm,sorted(pos)


def freeze():
    pp=OUT/'protocol.json'
    old=prior.read(source.OUT/'protocol.json');prior.verify(old)
    if pp.exists():p=prior.read(pp);prior.verify(p);checks(old,p);return p
    fit=prior.read(FIT/'protocol.json');check_fit(fit)
    summary=prior.read(FIT/'summary.json');prior.verify(summary)
    if summary['status']!='native_pool_fit_complete' or set(summary['groups'])!={f'{a}-{s}' for s in (7,17,27) for a in ('R','V','VM')}:raise ValueError('Incomplete fit diagnosis')
    review=prior.read(DEST/'review-summary.json');prior.verify(review)
    paths=[source.OUT/'protocol.json',source.OUT.parent/'dataset-completion.json',FIT/'protocol.json',FIT/'summary.json',DEST/'review-summary.json',Path(__file__).resolve()]
    for key in KEYS:source.complete(key);paths += [source.OUT/'training'/key/'completion.json',source.OUT/'evaluation'/f'{key}.json']
    # Reuse all 37 previously full-label reviewed sources. No new whole-pool approval.
    rp=source.OUT.parent/'reviewed-completion.json';r=prior.read(rp);prior.verify(r);paths.append(rp)
    if len(r['decisions'])!=113 or len(r['members'])!=37:raise ValueError('Full-label review incomplete')
    for m in old['pool_rows']:
        if Counter(t['class_name'] for t in truth_for(m))!=Counter(m['class_instances']):raise ValueError('Class supervision drift')
        paths += [Path(m['image_path']),Path(m['label_path'])]
    sdpath=prior.ROOT/'data/research/ml_training_recovery_v1/unified-hold-physical-lighting-control-v1/design.json'
    sd=prior.read(sdpath);prior.verify(sd);sf=Path(sd['solver_file'])
    if prior.file_sha256(sf)!=sd['inputs'][str(sf)]:raise ValueError('Solver changed')
    sys.path.insert(0,str(sf.parent.parent));from scipy.optimize import milp,Bounds,LinearConstraint
    paths += [sdpath,sf];OUT.mkdir(exist_ok=True)
    result={str(s):solve(old,s,milp,Bounds,LinearConstraint) for s in (7,17,27)}
    cp=OUT/'counts.json';prior.frozen(cp,dict(status='counts_checked',results=result,training_started=False,
        objective='Minimize maximum additional old-member decrement; then versioned-hash-rank linear tie objective',
        multiplier=3,new_image_exposures=162,total_image_exposures=2700,
        inputs={str(p):prior.file_sha256(p) for p in paths}));paths.append(cp)
    if any(w['status']!='exact_threefold_dose_witness' for w in result.values()):raise ValueError('Threefold fixed-budget design not feasible; no automatic relaxation')
    p={k:old[k] for k in ('pool_rows','names','initialization','evaluation','held_members','training_config','brightness_factors')}
    p.update(schedules={},listings={},ledger={},replacement_positions={})
    for seed in (7,17,27):
        v,vm,positions=schedules(old,seed,result[str(seed)])
        # Existing low-dose replacements plus added replacements; same per-position brightness.
        p['replacement_positions'][str(seed)]=sorted(set(old['replacement_positions'][str(seed)])|set(positions))
        for arm,seq in (('V',v),('VM',vm)):
            key=f'{arm}-{seed}';p['schedules'][key]=seq;lookup={m['member_id']:m for m in p['pool_rows']}
            listing=OUT/f'{key}.txt';listing.write_text(''.join(lookup[m]['image_path']+'\n' for m in sorted(set(seq))))
            p['listings'][key]=str(listing);paths.append(listing)
            windows=[]
            for start in range(0,2700,300):
                part=seq[start:start+300]
                windows.append(dict(first_step=start//6+1,image_exposures=dict(Counter(part)),
                    class_instances={c:sum(lookup[m]['class_instances'].get(c,0) for m in part) for c in CLASSES},
                    lineage_exposures=dict(Counter(lookup[m]['lineage_id'] for m in part))))
            p['ledger'][key]=windows
    checks(old,p)
    p.update(status='dose_sequences_frozen_preflight_pending',new_member_multiplier=3,
        interpretation='Fixed total and class budgets: replace more old bridge exposures with reviewed new poses; appearance pair controlled. Not pure repetition without opportunity cost.',
        historical_controls={k:str(source.OUT/'evaluation'/f'{k}.json') for k in KEYS},
        training_ready=False,training_started=False,independent_source_poses=13,material_pose_groups=12,
        limits=['Shared layout/assets; C01 and S07 concentration ratios unchanged','Old bridge exposure decreases; this is part of the strategy',
                'Member/brightness combinations change at 108 additional positions','Threefold is a bounded test, not a proven sufficient exposure threshold',
                'Development unknown-content instances stay in full metrics','Historical whole-pool quality not re-certified'],
        inputs={str(x):prior.file_sha256(x) for x in paths})
    return prior.frozen(pp,p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    if a.freeze:print(freeze()['status'])
    else:
        p=prior.read(OUT/'protocol.json');prior.verify(p);checks(prior.read(source.OUT/'protocol.json'),p)
        print('PREFLIGHT_ONLY_NO_TRAINING')


if __name__=='__main__':main()
