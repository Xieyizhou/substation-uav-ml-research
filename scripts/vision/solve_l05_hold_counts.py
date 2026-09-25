"""One frozen count solve for L05 hold; never trains or grants quality approval."""
import sys
from collections import Counter
from pathlib import Path
from scripts.vision.prepare_l05_compensation_review import OUT,prior,SCOPE
from scripts.vision.prepare_physical_lighting_control import SOURCE
from scripts.vision import optimize_revision_compensation as solver

def main():
    dest=OUT/'counts.json'
    if dest.exists():prior.verify(prior.read(dest));print('VALID_FROZEN_RESULT_REUSED');return
    pp=SOURCE/'protocol.json';ap=solver.prior.OUT/'counts.json';rp=solver.prior.POLICY/'protocol.json'
    paths=[pp,ap,rp,SCOPE/'scope.json',OUT/'evidence/manifest.json',Path(__file__).resolve(),Path(solver.__file__)]
    for x in paths[:5]:prior.verify(prior.read(x))
    p,a,policy=map(prior.read,(pp,ap,rp));scope=prior.read(SCOPE/'scope.json')
    rows=p['pool_rows'];held=set(p['held_members'])|set(scope['proposed_whole_frame_hold']);allow=set(a['reviewed_increase_candidates'])-held
    if len(a['reviewed_increase_candidates'])!=29:raise ValueError('Candidate scope changed')
    sf=next(Path(k) for k in policy['inputs'] if k.endswith('scipy/__init__.py'))
    if prior.file_sha256(sf)!=policy['inputs'][str(sf)]:raise ValueError('Solver identity changed')
    sys.path.insert(0,str(sf.parent.parent));paths.append(sf)
    results={};queue=set()
    for seed in (7,17,27):
        old=Counter(p['schedules'][f'brightness-450-{seed}'])
        result=solver.optimize(rows,old,held,allow,policy['risk_lineage_groups']);results[str(seed)]=result
        queue.update(c['member_id'] for c in result.get('changes',[]) if c['delta']>0)
        print(seed,{k:v for k,v in result.items() if k not in ('counts','changes')},flush=True)
    prior.frozen(dest,dict(status='count_solution_quality_pending' if all(r['status']=='lexicographic_integer_optimum_verified' for r in results.values()) else 'count_solution_not_established',
        results=results,held_members=sorted(held),allowlist=sorted(allow),increase_review_queue=sorted(queue),
        solve_round=1,max_solve_rounds=1,training_ready=False,training_started=False,
        constraints='Frozen existing 29 candidate scope; current zero exposures remain zero; risk group counts no increase; subset/full-class totals preserved; negatives fixed.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('INCREASE_REVIEW_QUEUE',sorted(queue))

if __name__=='__main__':main()
