"""Frozen reviewed-allowlist and lineage-capped exposure control."""
import argparse,hashlib,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.vision.risk_capped_second_review import CONTROL,OUT as SECOND
from scripts.vision.check_risk_capped_redistribution import RUN,ROOT,ready as old_ready,read,verify,frozen,file_sha256,solve,validate_capped
from scripts.vision.risk_capped_control_review import COUNTS
from scripts.vision.decide_risk_capped_control import validate
from scripts.vision.order_retention_runtime import preflight_cell,overrides,check_actual
from scripts.vision.exposure_protocol import exposures
from scripts.vision.prepare_whole_image_hold import baseline_verify

OUT=RUN/'lineage-capped-exposure-control-v1'
KEYS=tuple(f'lineage-capped-450-{s}' for s in (7,17,27))
VERSION='lineage-capped-exposure-control-v1'

def digest(*parts):return hashlib.sha256(':'.join(map(str,(VERSION,*parts))).encode()).hexdigest()

def sequence(rows,original,counts,seed):
    idx={r['member_id']:r for r in rows};old=Counter(original);result=list(original);holes={};extra={}
    for m in sorted(idx):
        if counts[m]<old[m]:
            positions=sorted((i for i,v in enumerate(original) if v==m),key=lambda i:(digest(seed,m,i),i))[:old[m]-counts[m]]
            holes.setdefault(idx[m]['subset'],[]).extend(positions)
        elif counts[m]>old[m]:extra.setdefault(idx[m]['subset'],{})[m]=counts[m]-old[m]
    for subset,slots in holes.items():
        remaining=extra.get(subset,{}).copy();stream=[];cycle=0
        while any(remaining.values()):
            for m in sorted((m for m,v in remaining.items() if v),key=lambda m:(digest(seed,subset,cycle,m),m)):
                stream.append(m);remaining[m]-=1
            cycle+=1
        if len(stream)!=len(slots):raise ValueError('Subset slots mismatch')
        for i,m in zip(sorted(slots),stream):result[i]=m
    if Counter(result)!=+Counter(counts):raise ValueError('Sequence count mismatch')
    for a,b in zip(original,result):
        if idx[a]['subset']!=idx[b]['subset'] or (idx[a]['subset']=='hard_negative' and a!=b):raise ValueError('Negative/subset position changed')
    return result

def check_policy(rows,old,new,held,caps,groups):
    validate_capped(rows,old,new,held,caps)
    for members in groups.values():
        if sum(new[m] for m in members)>sum(old.get(m,0) for m in members):raise ValueError('Risk lineage exposure increased')

def freeze():
    if (OUT/'protocol.json').exists():verify(read(OUT/'protocol.json'));return
    p=old_ready();idx={r['member_id']:r for r in p['pool_rows']};held=set(p['held_member_ids']);reviewed=set();bad=set()
    paths=[RUN/'protocol.json',RUN/'ready.json',CONTROL/'completion.json',COUNTS/'counts.json',Path(__file__),ROOT/'scripts/vision/check_risk_capped_redistribution.py',ROOT/'scripts/vision/order_retention_runtime.py',ROOT/'scripts/vision/exposure_protocol.py']
    for root in (CONTROL,SECOND):
        e=read(root/'evidence.json');r=read(root/'review.json');verify(e);verify(r);validate(e,r['decisions']);paths.extend([root/'evidence.json',root/'review.json'])
        for d in r['decisions']:
            reviewed.add(d['member_id'])
            if d['status']=='insufficient_or_uncertain':bad.add(d['member_id'])
    verify(read(CONTROL/'completion.json'));prior=read(COUNTS/'counts.json');verify(prior)
    risks=bad|set(prior['risk_members']);lineages={idx[m]['lineage_id'] for m in risks}
    groups={g:sorted(m for m,r in idx.items() if r['lineage_id']==g) for g in sorted(lineages)}
    safe=reviewed-bad-{m for ms in groups.values() for m in ms}
    caps={m for m,r in idx.items() if r['subset']!='hard_negative' and m not in safe|held}
    OUT.mkdir(exist_ok=True);schedules={};results={};listings={};ledger={}
    for seed in (7,17,27):
        original=p['schedules'][f'reference-450-{seed}'];old=Counter(original);new,obj=solve(p['pool_rows'],old,held,caps)
        if new is None:raise ValueError('Feasibility not established '+str(obj))
        check_policy(p['pool_rows'],old,new,held,caps,groups)
        key=f'lineage-capped-450-{seed}';seq=sequence(p['pool_rows'],original,new,seed);schedules[key]=seq;results[key]=dict(counts=new,objective=obj)
        listing=OUT/(key+'.txt');listing.write_text('\n'.join(idx[m]['image_path'] for m in sorted(set(seq)))+'\n');listings[key]=str(listing);paths.append(listing)
        ledger[key]=dict(total=exposures(p['pool_rows'],seq),windows_50_steps=[exposures(p['pool_rows'],seq[i:i+300]) for i in range(0,2700,300)])
    for r in p['pool_rows']:
        for k in ('image','label'):
            path=Path(r[k+'_path'])
            if file_sha256(path)!=r[k+'_sha256']:raise ValueError('Stale member')
            paths.append(path)
    import scipy
    paths.append(Path(scipy.__file__))
    frozen(OUT/'protocol.json',dict(status='frozen_preflight_pending',pool_rows=p['pool_rows'],schedules=schedules,listings=listings,ledger=ledger,counts=results,
        names=p['names'],initialization=p['initialization'],controls=p['controls'],evaluation=p['evaluation'],acceptance_policy=p['acceptance_policy'],retention=p['retention'],
        held_member_ids=sorted(held),increase_allowlist=sorted(safe),capped_members=sorted(caps),risk_lineage_groups=groups,
        known_risk_members=sorted(risks),lineage_limit='Registered groups only; member-placeholder identities are not proof of independent provenance.',
        policy='Only explicitly reviewed non-risk members may increase. Unknown/unreviewed positives retain a reference-count ceiling, not a negative quality verdict.',
        interpretation='Joint member/lineage-limited compensation policy; not pure causal effect of total class exposure.',
        training_started=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('FROZEN31_REVIEWED_ALLOWLIST',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p)
    for key in KEYS:
        dest=OUT/'preflight'/key/'completion.json';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():verify(read(dest));check_actual(p,key,read(dest)['actual']);continue
        try:u=preflight_cell(p,key)
        except BaseException:
            frozen(dest.parent/'failure.json',dict(error=traceback.format_exc()));raise
        frozen(dest,dict(**u,protocol_identity=p['identity'],inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}));print('LOADER_PASS',key,flush=True)
    import subprocess
    tests=['tests.test_lineage_capped_control','tests.test_risk_capped_control_review','tests.test_risk_capped_redistribution','tests.test_whole_image_hold_train']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Pinned baseline failure')
    paths=[OUT/'protocol.json',Path(__file__)]+[OUT/'preflight'/k/'completion.json' for k in KEYS]+[ROOT/(x.replace('.','/')+'.py') for x in tests]
    frozen(OUT/'ready.json',dict(status='ready_for_training_not_started',cells=list(KEYS),regression_output=t.stderr,baseline=baseline,whole_repository_tested=False,
        unresolved_historical_risks_retained_at_no_increase=True,inputs={str(x):file_sha256(x) for x in paths}))
    print('READY_THREE_SEEDS_NO_TRAINING',flush=True)

def ready():
    r=read(OUT/'ready.json');p=read(OUT/'protocol.json');verify(r);verify(p)
    if r['status']!='ready_for_training_not_started' or set(r['cells'])!=set(KEYS):raise ValueError('Not ready')
    old=old_ready()
    for key in KEYS:
        u=read(OUT/'preflight'/key/'completion.json');verify(u);check_actual(p,key,u['actual'])
        if u['protocol_identity']!=p['identity'] or any(u[x] is not False for x in ['optimizer_created','backward_executed','validation_run']):raise ValueError('Invalid loader receipt')
        seed=int(key.split('-')[-1]);orig=old['schedules'][f'reference-450-{seed}'];counts=p['counts'][key]['counts']
        check_policy(p['pool_rows'],Counter(orig),counts,set(p['held_member_ids']),set(p['capped_members']),p['risk_lineage_groups'])
        if sequence(p['pool_rows'],orig,counts,seed)!=p['schedules'][key]:raise ValueError('Sequence changed')
    if file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Initial weight changed')
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--preflight',action='store_true');ap.add_argument('--solver-path');a=ap.parse_args()
    if a.freeze:
        if not a.solver_path:ap.error('Explicit isolated solver path required')
        sys.path.insert(0,a.solver_path);freeze()
    elif a.preflight:preflight()
    else:print('READ_ONLY_NO_TRAINING')
