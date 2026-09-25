"""One paired physical-lighting tail experiment; no training in default command."""
import copy
from collections import Counter
from hashlib import sha256
from pathlib import Path
from scripts.vision.physical_low_light_capture import OUT,freeze as capture_design,prior
from scripts.vision.physical_low_light_dataset import export
from scripts.vision.closed_budget_design import freeze as base_design,SOURCE as BASE_PARENT,OUT as BASE
from scripts.vision.freeze_closed_source_control import counts
from scripts.vision.train_closed_budget import complete

KEYS=tuple(f'{family}-{seed}' for seed in (7,17,27) for family in ('R1000','L1000'))
VERSION='physical-low-light-control-v1'
def digest(*args):return sha256('|'.join(map(str,(VERSION,)+args)).encode()).hexdigest()

def tails(seed,rows):
    # 300 source tokens, exactly 75 per planned class; every token forms two intact batches.
    tokens=[]
    for category in ('transformer','switchgear','capacitor_bank','reactor'):
        members=sorted((r for r in rows if r['planned_category']==category),key=lambda r:digest(seed,category,r['member_id']))
        if len(members)!=8:raise ValueError('Incomplete class/layout/material source matrix')
        for i in range(75):tokens.append((members[i%8]['source_member_id'],members[i%8]['member_id'],i,category))
    tokens.sort(key=lambda t:digest(seed,*t))
    ref=[];treatment=[];batches=[]
    for pair in range(50):
        group=tokens[pair*6:pair*6+6];original=[t[0] for t in group];low=[t[1] for t in group]
        order=(0,1) if int(digest(seed,'condition',pair),16)%2==0 else (1,0)
        for flag in order:
            ref.extend(original);treatment.extend(low if flag else original);batches.append(dict(step=901+len(batches),pair=pair,low_light=bool(flag),source_members=original))
    return ref,treatment,batches

def check(p,old,new):
    lookup={m['member_id']:m for m in p['pool_rows']}
    if len(lookup)!=len(p['pool_rows']):raise ValueError('Duplicate members')
    for m in lookup.values():
        for f in ('image','label'):
            if prior.file_sha256(m[f+'_path'])!=m[f+'_sha256']:raise ValueError('Member identity drift')
    for seed in (7,17,27):
        a,b,batches=tails(seed,new);ref=f'B900-{seed}';expected_config={**old['training_config'][ref],'epochs':100}
        for fam,tail in (('R1000',a),('L1000',b)):
            key=f'{fam}-{seed}';seq=p['schedules'][key]
            if seq!=old['schedules'][ref]+tail or len(seq)!=6000:raise ValueError('Exposure sequence drift')
            if p['brightness_factors'][key]!=old['brightness_factors'][ref]+[1.]*600 or p['training_config'][key]!=expected_config:raise ValueError('Training config/brightness drift')
            if any(m in p['held_members'] for m in seq):raise ValueError('Held member restored')
            if p['tail_batches'][key]!=batches or p['totals'][key]!=counts(seq,lookup):raise ValueError('Tail/ledger mismatch')
        r=p['schedules'][f'R1000-{seed}'];l=p['schedules'][f'L1000-{seed}']
        for x,y in zip(r,l,strict=True):
            if Path(lookup[x]['label_path']).read_bytes()!=Path(lookup[y]['label_path']).read_bytes() or lookup[x]['lineage_id']!=lookup[y]['lineage_id']:raise ValueError('Paired supervision/lineage drift')
        if sum(x!=y for x,y in zip(r,l))!=300:raise ValueError('Wrong lighting treatment dose')
        if sum(lookup[m]['subset']=='hard_negative' for m in r)!=972:raise ValueError('Negative exposure changed')
        if Counter(lookup[m]['source_member_id'] for m in b if m.startswith('low-'))!=Counter(m for m in b if not m.startswith('low-')):raise ValueError('Tail light/normal imbalance')

def freeze():
    capture_design();manifest=export();old=base_design();dest=OUT/'design.json';new=manifest['members']
    if dest.exists():p=prior.read(dest);prior.verify(p);check(p,old,new);return p
    p={k:copy.deepcopy(old[k]) for k in ('pool_rows','names','initialization','evaluation','held_members','acceptance_policy','environment')}
    p['pool_rows']+=new;p.update(schedules={},brightness_factors={},training_config={},listings={},totals={},ledger={},tail_batches={})
    lookup={m['member_id']:m for m in p['pool_rows']};deps=[OUT/'capture-protocol.json',OUT/'quality-review.json',OUT/'export/manifest.json',BASE/'design.json',Path(__file__).resolve()]
    for seed in (7,17,27):
        ref=f'B900-{seed}';complete(ref);deps.append(BASE/'training'/ref/'completion.json');a,b,batches=tails(seed,new)
        for family,tail in (('R1000',a),('L1000',b)):
            key=f'{family}-{seed}';seq=old['schedules'][ref]+tail;p['schedules'][key]=seq;p['brightness_factors'][key]=old['brightness_factors'][ref]+[1.]*600
            p['training_config'][key]={**old['training_config'][ref],'epochs':100};p['tail_batches'][key]=batches
            listing=OUT/f'{key}.txt';text=''.join(lookup[m]['image_path']+'\n' for m in sorted(set(seq)))
            if listing.exists():
                if listing.read_text()!=text:raise ValueError('Listing mismatch')
            else:listing.write_text(text)
            p['listings'][key]=str(listing);deps.append(listing);p['totals'][key]=counts(seq,lookup)
            p['ledger'][key]=[dict(first_step=i//6+1,**counts(seq[i:i+300],lookup)) for i in range(0,6000,300)]
    p.update(status='paired_physical_light_design_frozen_preflight_pending',
        comparison='R1000 vs L1000: exact same 900-step B900 prefix, then 100 source-rehearsal steps; only 50 complete tail batches change to actual lower-light counterpart pixels. Both independent initializations.',
        interpretation='This tests physical-lighting tail strategy, not all lighting diversity. B900-to-R1000 also changes source exposure and total steps and is not a pure steps effect.',
        source_independence='8 existing training poses, 2 layouts, shared assets; 32 relit frames are derivatives, not 32 independent scenes.',
        controls='Negative members and positions unchanged. Same full class instance supervision and lineage counts between paired arms. Gamma disabled. No threshold search.',
        training_started=False,inputs={str(p):prior.file_sha256(p) for p in deps})
    check(p,old,new);return prior.frozen(dest,p)

if __name__=='__main__':print(freeze()['status'])
