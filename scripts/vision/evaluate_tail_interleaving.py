"""Adapt the fixed evaluator in this process only; historical modules unchanged."""
import argparse
from pathlib import Path
from scripts.vision.train_tail_interleaving import OUT,SOURCE,prior,contract as training_contract
from scripts.vision import evaluate_physical_low_light as fixed
KEYS=tuple(f'I1000-{s}' for s in (7,17,27))

def protocol():
    path=OUT/'design.json'
    if path.exists():p=prior.read(path);prior.verify(p);return p
    p,_,deps,_=training_contract();p.pop('identity',None)
    p['inputs']={str(d):prior.file_sha256(d) for d in deps}
    p['status']='interleaved_evaluation_protocol';return prior.frozen(path,p)

def complete(key):
    if key not in KEYS:raise ValueError('Unknown unit')
    p=protocol();cp=OUT/'training'/key/'completion.json';c=prior.read(cp);prior.verify(c)
    x=prior.read(c['exposure_path']);prior.verify(x)
    if c['optimizer_steps']!=1000 or x['optimizer_steps']!=1000 or x['actual']!=p['schedules'][key]:raise ValueError('Incomplete actual exposure')
    from scripts.vision.closed_budget_runtime import check
    check(p,key,x['actual'],x['brightness_log'])
    if prior.file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Endpoint hash changed')
    return c

def evaluate(key):
    binding=OUT/'evaluation'/f'{key}-adapter-binding.json'
    if binding.exists():prior.verify(prior.read(binding))
    fixed.OUT=OUT;fixed.KEYS=KEYS;fixed.complete=complete;fixed.contract=lambda key:(protocol(),protocol(),None)
    result=fixed.evaluate(key)
    if not binding.exists():
        deps=[Path(__file__).resolve(),Path(fixed.__file__).resolve(),OUT/'evaluation'/f'{key}.json']
        prior.frozen(binding,dict(status='adapter_identity_bound',inputs={str(d):prior.file_sha256(d) for d in deps}))
    return result

def finish():
    records=[];refs=[];deps=[OUT/'design.json',Path(__file__).resolve()]
    for key in KEYS:
        records.append(evaluate(key));deps.extend([OUT/'evaluation'/f'{key}.json',OUT/'evaluation'/f'{key}-adapter-binding.json'])
        rp=SOURCE/'evaluation'/f"R1000-{key.split('-')[-1]}.json";r=prior.read(rp);prior.verify(r);refs.append(r);deps.append(rp)
    g=fixed.aggregate(records);ref=fixed.aggregate(refs);p=protocol()
    oldp=SOURCE/'evaluation/summary.json';old=prior.read(oldp);prior.verify(old);deps.append(oldp)
    historical_paths=[fixed.PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hpath=Path(p['evaluation']['historical_reference']);h=prior.read(hpath);prior.verify(h)
    historical=[prior.read(x) for x in historical_paths]
    for x in historical:prior.verify(x)
    _,policy,_=fixed.reference_contract('fixed-7');deps+=historical_paths+[hpath]
    b=fixed.baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    return prior.frozen(OUT/'evaluation/summary.json',dict(status='numerical_complete_explicit_review_pending',aggregate=g,
        R1000=ref,B900=old['direct_reference_aggregate'],policy=fixed.policy_checks(g,fixed.aggregate(historical),h['historical_A'],policy),
        relative_R1000=fixed.direct_checks(g,ref),relative_B900=fixed.direct_checks(g,old['direct_reference_aggregate']),
        paired_changes={str(s):fixed.paired_change(a['rows'],z['rows']) for s,a,z in zip((7,17,27),refs,records)},
        matching_conflicts=sum(r['matching_conflicts'] for r in records),baseline=b,selected_candidate=None,
        inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=KEYS);ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    if a.evaluate:
        if a.cell:evaluate(a.cell)
        else:finish()
    else:protocol();print('PROTOCOL_ONLY_NO_INFERENCE')
