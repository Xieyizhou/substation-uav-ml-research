"""Freeze and solve the authorized revised design once; no capture or training."""
import argparse,sys
from collections import Counter
from pathlib import Path
from scripts.vision.validate_boundary_roundoff import OUT as POLICY,prior
from scripts.vision.prepare_physical_lighting_control import SOURCE,OUT as LIGHT
from scripts.vision import optimize_revision_compensation as solver
from scripts.vision.finalize_l05_hold_block import validate_counts

OUT=POLICY.parent/'unified-hold-physical-lighting-control-v1'

def freeze():
    dest=OUT/'design.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    paths=[POLICY/'completion.json',POLICY/'policy.json',POLICY/'quality-precheck.json',SOURCE/'protocol.json',LIGHT/'protocol.json',solver.prior.POLICY/'protocol.json']
    for p in paths:prior.verify(prior.read(p))
    policy=prior.read(paths[1]);quality=prior.read(paths[2]);source=prior.read(paths[3]);risk=prior.read(paths[5])
    held=policy['future_zero_exposure_members'];allow=quality['eligible_members']
    if len(held)!=10 or len(allow)!=27 or set(held)&set(allow):raise ValueError('Frozen scope conflict')
    reviewed,extra=solver.prior.reviewed_members(source['pool_rows']);paths+=extra
    if not set(allow)<=reviewed:raise ValueError('Full-label review changed')
    sf=next(Path(k) for k in risk['inputs'] if k.endswith('scipy/__init__.py'))
    if prior.file_sha256(sf)!=risk['inputs'][str(sf)]:raise ValueError('Solver changed')
    paths += [sf,Path(__file__).resolve(),Path(solver.__file__),prior.ROOT/'docs/results/ml_unified_hold_physical_lighting_design_20260910.md']
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='design_frozen_counts_not_solved',held_members=held,allowlist=allow,pool_rows=source['pool_rows'],
        original_sequences=source['schedules'],brightness_factors=source['brightness_factors'],risk_groups=risk['risk_lineage_groups'],
        initialization=source['initialization'],evaluation=source['evaluation'],acceptance_policy=source['acceptance_policy'],retention=source['retention'],
        learning_rate=.0005,device='cpu',steps=450,batch=6,max_solve_rounds=1,solver_file=str(sf),
        light_sources=prior.read(LIGHT/'protocol.json')['selected'][:4],light_quota_caps=[21,8,23,23],
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))

def run():
    p=freeze();sys.path.insert(0,str(Path(p['solver_file']).parent.parent));paths=[OUT/'design.json'];results={}
    for seed in (7,17,27):
        folder=OUT/f'count-seed-{seed}';dest=folder/'result.json'
        if dest.exists():result=prior.read(dest);prior.verify(result);results[str(seed)]=result['result'];paths.append(dest);continue
        if folder.exists():raise ValueError('Incomplete solve retained; no automatic new solve')
        folder.mkdir();prior.frozen(folder/'started.json',dict(seed=seed,solve_round=1,inputs={str(OUT/'design.json'):prior.file_sha256(OUT/'design.json')}))
        old=Counter(p['original_sequences'][f'brightness-450-{seed}'])
        result=solver.optimize(p['pool_rows'],old,set(p['held_members']),set(p['allowlist']),p['risk_groups'])
        if result['status']=='lexicographic_integer_optimum_verified':validate_counts(p['pool_rows'],old,result['counts'],set(p['held_members']),set(p['allowlist']),p['risk_groups'])
        prior.frozen(dest,dict(result=result,seed=seed,inputs={str(x):prior.file_sha256(x) for x in (OUT/'design.json',folder/'started.json')}));paths.append(dest);results[str(seed)]=result
        print(seed,{k:v for k,v in result.items() if k not in ('counts','changes')},flush=True)
    good=all(x['status']=='lexicographic_integer_optimum_verified' for x in results.values());quotas={}
    if good:
        quotas={f['member_id']:min(cap,min(r['counts'][f['member_id']] for r in results.values())//2) for cap,f in zip(p['light_quota_caps'],p['light_sources'])}
    status='counts_verified_capture_pending' if good and all(quotas.values()) else 'design_blocked'
    prior.frozen(OUT/'counts.json',dict(status=status,results=results,light_quotas=quotas,training_ready=False,training_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(status,quotas)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--solve',action='store_true');a=ap.parse_args()
    if a.solve:run()
    else:freeze();print('DESIGN_FROZEN_NO_SOLVE')
