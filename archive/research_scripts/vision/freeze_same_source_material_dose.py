"""Exact same-source fifteen-slot material replacement, no score selection."""
import copy
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from scripts.vision.same_source_material_quality_v2 import OUT,prior,reference,main as quality
from scripts.vision.audit_material_dose_feasibility import capacity

KEYS=('D-7','D-17','D-27')
COLORS=('warm','cool','gray_target_body')
VERSION='same-source-material-dose-control-v1'

def allocate(slots,seed):
    sources=sorted({s for _,s in slots})
    order=sorted(slots,key=lambda x:sha256(f'{VERSION}|{seed}|{x[1]}|{x[0]}'.encode()).hexdigest())
    @lru_cache(None)
    def search(i,totals,source_counts):
        if i==len(order):
            if totals!=(5,5,5):return None
            if any(max(source_counts[j:j+3])-min(source_counts[j:j+3])>1 for j in range(0,len(source_counts),3)):return None
            return ()
        pos,sid=order[i];offset=sources.index(sid)*3
        for color in sorted(range(3),key=lambda c:sha256(f'{VERSION}|{seed}|{sid}|{pos}|{COLORS[c]}'.encode()).hexdigest()):
            if totals[color]>=5:continue
            t=list(totals);sc=list(source_counts);t[color]+=1;sc[offset+color]+=1
            if sc[offset+color]>(sum(s==sid for _,s in slots)+2)//3:continue
            result=search(i+1,tuple(t),tuple(sc))
            if result is not None:return (color,)+result
        return None
    result=search(0,(0,0,0),(0,)*(3*len(sources)))
    if result is None:raise ValueError('Exact balanced assignment infeasible')
    return {pos:COLORS[c] for (pos,_),c in zip(order,result,strict=True)}

def check(p,old):
    rows={r['member_id']:r for r in p['pool_rows']}
    for seed in (7,17,27):
        key=f'D-{seed}';before=old['schedules'][f'T-{seed}'];after=p['schedules'][key]
        changes=[i for i,(a,b) in enumerate(zip(before,after,strict=True)) if a!=b]
        if len(after)!=2700 or changes!=p['changed_positions'][key] or len(changes)!=15:raise ValueError('Changed position budget')
        if p['brightness_factors'][key]!=old['brightness_factors'][f'T-{seed}'] or p['training_config'][key]!=old['training_config'][f'T-{seed}']:raise ValueError('Config/brightness changed')
        for i,(a,b) in enumerate(zip(before,after,strict=True)):
            if a==b:continue
            x,y=rows[a],rows[b]
            if x.get('variant')!='original' or y.get('variant') not in COLORS or x['pair_id']!=y['pair_id'] or x['subset']!=y['subset']:raise ValueError('Illegal replacement')
            if x['class_instances']!=y['class_instances'] or Path(x['label_path']).read_bytes()!=Path(y['label_path']).read_bytes():raise ValueError('Complete label drift')
        if Counter(rows[after[i]]['variant'] for i in changes)!=Counter({c:5 for c in COLORS}):raise ValueError('Color quota drift')
        if any(m in p['held_members'] for m in after):raise ValueError('Held member restored')

def protocol():
    q=quality()
    if q['gaps']:raise ValueError('Quality binding gaps')
    old,_,_=reference.contract('T-7')
    p={k:copy.deepcopy(old[k]) for k in ('pool_rows','names','initialization','evaluation','held_members')}
    p.update(schedules={},brightness_factors={},training_config={},listings={},changed_positions={},ledger={})
    rows={r['member_id']:r for r in p['pool_rows']};variants={(r.get('pair_id'),r.get('variant')):r['member_id'] for r in p['pool_rows'] if 'full_truth' in r}
    paths=[OUT/'quality-v2.json',reference.OUT/'protocol.json',Path(__file__).resolve()]
    for seed in (7,17,27):
        ref=f'T-{seed}';key=f'D-{seed}';reference.complete(ref)
        cap=capacity(old,ref);slots=[(i,x['pair_id']) for x in cap['eligible'] for i in x['positions']]
        if len(slots)!=15:raise ValueError('Capacity drift')
        assignment=allocate(slots,seed);seq=list(old['schedules'][ref])
        for i,pair in slots:seq[i]=variants[pair,assignment[i]]
        p['schedules'][key]=seq;p['brightness_factors'][key]=old['brightness_factors'][ref];p['training_config'][key]=old['training_config'][ref]
        p['changed_positions'][key]=sorted(assignment)
        # Listing may retain zero-exposure originals only if loader excludes them: write the exact member set.
        listing=OUT/(key+'.txt');content=''.join(rows[m]['image_path']+'\n' for m in sorted(set(seq)))
        if listing.exists():
            if listing.read_text()!=content:raise ValueError('Listing drift')
        else:listing.write_text(content)
        p['listings'][key]=str(listing);paths.append(listing)
        p['ledger'][key]=[dict(first_step=i//6+1,image_exposures=dict(Counter(seq[i:i+300])),
            classes={c:sum(rows[m]['class_instances'].get(c,0) for m in seq[i:i+300]) for c in p['names']},
            lineages=dict(Counter(rows[m]['lineage_id'] for m in seq[i:i+300]))) for i in range(0,2700,300)]
    p.update(status='frozen_semantic_authorization_pending',training_started=False,
        acceptance_policy=prior.read(reference.OUT/'decision-protocol.json'),
        inputs={str(x):prior.file_sha256(x) for x in paths})
    check(p,old);dest=OUT/'protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);check(r,old);return r
    return prior.frozen(dest,p)

if __name__=='__main__':print(protocol()['status'])
