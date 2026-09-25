"""Exact deterministic V/VM sequences; no training entry or readiness claim."""
import hashlib
from collections import Counter
from pathlib import Path
from scripts.vision.merge_compensated_material_candidates import OUT as DATA,prior
from scripts.vision.merge_material_pose_candidates import OUT as ORIGINAL
from scripts.vision.train_frozen_multiscale import contract

OUT=DATA/'sequence-freeze-v1'
VERSION='compensated-material-sequence-v1'


def rank(*parts):return hashlib.sha256(':'.join(map(str,(VERSION,*parts))).encode()).hexdigest()


def replace(seq,removed,inserted,seed):
    positions=[]
    for mid,n in sorted(removed.items()):
        choices=sorted((i for i,x in enumerate(seq) if x==mid),key=lambda i:rank(seed,mid,i))
        if n>len(choices):raise ValueError('Excess removal')
        positions+=choices[:n]
    items=[]
    for cycle in range(max(inserted.values())):
        items += sorted((m for m,n in inserted.items() if n>cycle),key=lambda m:rank(seed,m,cycle))
    if len(positions)!=len(items):raise ValueError('Unequal replacement counts')
    out=list(seq)
    for i,m in zip(sorted(positions),items):out[i]=m
    return out,sorted(positions)


def main():
    mp=DATA/'candidate-export-v1/manifest.json';m=prior.read(mp);prior.verify(m)
    cp=ORIGINAL/'compensated-exposure-feasibility-v1/result.json';counts=prior.read(cp);prior.verify(counts)
    if len(m['members'])!=37 or m['reference_overlap_gaps']:raise ValueError('Candidate export gate')
    OUT.mkdir(exist_ok=True);schedules={};brightness={};configs={};listings={};ledger={};changes={};paths=[mp,cp,Path(__file__)]
    source=None;pool=None
    for seed in (7,17,27):
        design,s,actual=contract(f'fixed-{seed}');source=s
        rows={r['member_id']:r for r in s['pool_rows']}
        for x in m['members']:rows[x['member_id']]=dict(x,subset='bridge_positive',lineage_id=x['pair_id'])
        pool=list(rows.values());r=counts['results'][str(seed)]
        if r['status']!='exact_replacement_counts_feasible_not_training_ready':raise ValueError('No exact solution')
        old=actual['actual'];v,positions=replace(old,r['removed'],r['inserted'],seed);vm=list(v)
        for mid in r['inserted']:
            if mid=='C01-original':continue
            slots=[i for i,x in enumerate(v) if x==mid]
            variants=sorted(('original','warm','cool'),key=lambda a:rank(seed,mid,a))
            for j,i in enumerate(slots):vm[i]=mid.removesuffix('-original')+'-'+variants[j%3]
        for arm,seq in (('V',v),('VM',vm)):
            key=f'{arm}-{seed}';schedules[key]=seq;brightness[key]=s['brightness_factors'][f'R-clean-{seed}']
            configs[key]=s['training_config'][f'R-clean-{seed}']
            if len(seq)!=2700:raise ValueError('Budget changed')
            if Counter(rows[x]['subset'] for x in seq)!=Counter(rows[x]['subset'] for x in old):raise ValueError('Subset changed')
            for cls in ('transformer','switchgear','capacitor_bank','reactor'):
                if sum(rows[x]['class_instances'].get(cls,0) for x in seq)!=sum(rows[x]['class_instances'].get(cls,0) for x in old):raise ValueError('Class exposure changed')
            if any(rows[old[i]]['subset']!='bridge_positive' and seq[i]!=old[i] for i in range(2700)):raise ValueError('Nonbridge position changed')
            if set(seq)&set(s['held_members']):raise ValueError('Held member restored')
            windows=[]
            for start in range(0,2700,300):
                part=seq[start:start+300]
                windows.append(dict(first_step=start//6+1,image_exposures=dict(Counter(part)),
                    class_instances={c:sum(rows[x]['class_instances'].get(c,0) for x in part) for c in ('transformer','switchgear','capacitor_bank','reactor')},
                    lineage_exposures=dict(Counter(rows[x]['lineage_id'] for x in part))))
            ledger[key]=windows;lp=OUT/(key+'.txt');text=''.join(rows[x]['image_path']+'\n' for x in sorted(set(seq)))
            if lp.exists() and lp.read_text()!=text:raise ValueError('Listing changed')
            if not lp.exists():lp.write_text(text)
            listings[key]=str(lp);paths.append(lp)
        if any(rows[a]['full_truth']!=rows[b]['full_truth'] for a,b in zip(v,vm) if a!=b):raise ValueError('Pair full labels changed')
        changes[str(seed)]=positions;paths.append(Path(design['source_protocol']))
    prior.frozen(OUT/'protocol.json',dict(status='sequences_frozen_loader_not_checked',pool_rows=pool,schedules=schedules,
        brightness_factors=brightness,training_config=configs,listings=listings,names=source['names'],initialization=source['initialization'],
        evaluation=source['evaluation'],held_members=source['held_members'],ledger=ledger,replacement_positions=changes,
        training_ready=False,training_started=False,source_role_final_audit_complete=False,
        common_compensation_member='C01-original',interpretation='V/VM same positions, full labels, fixed brightness; C01 unchanged common original. Existing pool members only decrease.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('SIX_SEQUENCES2700_FROZEN; NO_TRAINING')


if __name__=='__main__':main()
