"""Pair exposed truth identities within each seed; no cross-scene identity inference."""
from collections import Counter
from pathlib import Path
from scripts.vision.summarize_closed_budget_fit import main as summarize,OUT,prior
from scripts.vision.diagnose_closed_budget_fit import freeze,validate,KEYS,TRAIN

def compare(a,b,exposures):
    aa={r['member_id']:r for r in a};bb={r['member_id']:r for r in b}
    if len(aa)!=len(a) or set(aa)!=set(bb):raise ValueError('Changed member population')
    events=[]
    for member,old in aa.items():
        new=bb[member]
        if old['truth']!=new['truth']:raise ValueError('Full truth changed')
        if not exposures.get(member,0):continue
        old_hits={m['truth_index'] for m in old['matches']};new_hits={m['truth_index'] for m in new['matches']}
        for i,t in enumerate(old['truth']):
            before=i in old_hits;after=i in new_hits
            events.append(dict(member_id=member,truth=t,before_hit=before,after_hit=after,state='retained_hit' if before and after else 'loss' if before else 'gain' if after else 'retained_miss',
                exposures_before=exposures[member],exposures_after=new['exposures'],miss=next((m for m in new['misses'] if m['truth_index']==i),None)))
    return events

def main():
    p=freeze();summarize();dest=OUT/'paired-analysis.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',OUT/'summary.json',TRAIN/'evaluation-completion.json',Path(__file__).resolve()];models={}
    for key in KEYS:
        ep=OUT/(key+'.json');r=prior.read(ep);validate(r,p,key);models[key]=r;paths.append(ep)
    members={r['member_id']:r for r in p['members']};by_seed={};all_events=[]
    for seed in (7,17,27):
        a=f'B450-{seed}';b=f'B900-{seed}'
        if set(p['exposures'][a])!=set(p['exposures'][b]) or any(p['exposures'][b][m]!=2*n for m,n in p['exposures'][a].items()):raise ValueError('Exposure not exact doubled')
        events=compare(models[a]['rows'],models[b]['rows'],p['exposures'][a]);all_events.extend(dict(seed=seed,**e) for e in events)
        by_seed[str(seed)]={}
        for category in ('transformer','switchgear','capacitor_bank','reactor'):
            by_seed[str(seed)][category]={}
            for scope in ('all','original','material'):
                selected=[e for e in events if e['truth']['class_name']==category and (scope=='all' or (members[e['member_id']]['variant']=='original')==(scope=='original'))]
                by_seed[str(seed)][category][scope]=dict(instances=len(selected),before=sum(e['before_hit'] for e in selected),after=sum(e['after_hit'] for e in selected),states=dict(Counter(e['state'] for e in selected)))
    return prior.frozen(dest,dict(status='paired_training_fit_diagnosis_not_generalization',by_seed=by_seed,events=all_events,
        reactor_losses=[e for e in all_events if e['truth']['class_name']=='reactor' and e['state']=='loss'],
        limits=['Member/label-row identities used only within same training image; no train/development cross-scene instance equivalence inferred.',
                'Unexposed members excluded from fit counts but retained in model prediction outputs.',
                'Full-label fit does not certify that all labelled content is unoccluded; seed/variant repeats are not independent scenes.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=main()
    for seed,v in r['by_seed'].items():print(seed,v)
