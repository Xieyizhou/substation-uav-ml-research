"""The single authorized additional risk-cap solve; never trains."""
import argparse,sys
from collections import Counter
from pathlib import Path
from scripts.vision.risk_capped_control_review import OUT,COUNTS,RUN,ROOT,ready,read,verify,frozen,file_sha256
from scripts.vision.check_risk_capped_redistribution import solve,validate_capped

def main():
    dest=OUT/'resolve-01.json'
    if dest.exists():verify(read(dest));print('SINGLE_RESOLVE_ALREADY_RECORDED');return
    p=ready();c=read(COUNTS/'counts.json');r=read(OUT/'review.json');e=read(OUT/'evidence.json')
    for x in (c,r,e):verify(x)
    from scripts.vision.decide_risk_capped_control import validate
    validate(e,r['decisions'])
    risks=set(c['risk_members'])|set(r['new_risk_members']);results={}
    for seed in (7,17,27):
        old=Counter(p['schedules'][f'reference-450-{seed}']);new,obj=solve(p['pool_rows'],old,set(p['held_member_ids']),risks)
        results[str(seed)]=dict(counts=new,objective=obj)
        print('RESOLVE',seed,obj,flush=True)
    import scipy
    frozen(dest,dict(status='counts_feasible_review_required' if all(x['counts'] is not None for x in results.values()) else 'training_blocked_feasibility_not_established',
        results=results,risk_members=sorted(risks),additional_resolves_used=1,additional_resolves_remaining=0,training_started=False,
        inputs={str(x):file_sha256(x) for x in [COUNTS/'counts.json',OUT/'review.json',OUT/'evidence.json',Path(__file__),ROOT/'scripts/vision/check_risk_capped_redistribution.py',Path(scipy.__file__)]}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--solver-path');a=ap.parse_args()
    if a.solver_path:sys.path.insert(0,a.solver_path);main()
    else:print('PREFLIGHT_ONLY_NO_SOLVE_NO_TRAINING')
