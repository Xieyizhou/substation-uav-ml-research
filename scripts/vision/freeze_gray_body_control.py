"""Freeze six real file sequences; no training entry."""
from collections import Counter
from pathlib import Path
from scripts.vision.prepare_gray_body_training_candidates import OUT,prior
from scripts.vision import train_compensated_material as source
from scripts.vision.structure_fit import truth_for


def checks(p,old):
    rows={r['member_id']:r for r in p['pool_rows']}
    for seed in (7,17,27):
        r,g=f'R-{seed}',f'G-{seed}';a,b=p['schedules'][r],p['schedules'][g]
        if a!=old['schedules'][f'VM-{seed}'] or len(b)!=2700:raise ValueError('Reference/budget drift')
        for key in (r,g):
            if p['brightness_factors'][key]!=old['brightness_factors'][f'VM-{seed}'] or p['training_config'][key]!=old['training_config'][f'VM-{seed}']:
                raise ValueError('Configuration/brightness drift')
        changed=[]
        for i,(x,y) in enumerate(zip(a,b,strict=True)):
            if x==y:continue
            changed.append(i)
            if rows[x].get('variant') not in ('warm','cool') or rows[y].get('variant')!='gray_target_body':raise ValueError('Invalid replacement')
            if rows[x]['pair_id']!=rows[y]['pair_id'] or rows[x]['class_instances']!=rows[y]['class_instances']:raise ValueError('Unpaired source/classes')
            if Path(rows[x]['label_path']).read_bytes()!=Path(rows[y]['label_path']).read_bytes():raise ValueError('Full labels differ')
        if changed!=p['replacement_positions'][str(seed)] or len(changed)!=30:raise ValueError('Replacement positions differ')
        if Counter(rows[x]['subset'] for x in a)!=Counter(rows[x]['subset'] for x in b):raise ValueError('Subset budget drift')
        if set(b)&set(p['held_members']):raise ValueError('Held members restored')
        for c in p['names']:
            if sum(rows[x]['class_instances'].get(c,0) for x in a)!=sum(rows[x]['class_instances'].get(c,0) for x in b):raise ValueError('Class exposure drift')
    for row in rows.values():
        for k in ('image','label'):
            if prior.file_sha256(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('File drift')


def freeze():
    pp=source.OUT/'protocol.json';old=prior.read(pp);prior.verify(old)
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);checks(p,old);return p
    ep=OUT/'candidate-export-v1/manifest.json';e=prior.read(ep);prior.verify(e)
    if len(e['members'])!=12 or e['reference_overlap_gaps']:raise ValueError('Candidate exclusion gate')
    rows=list(old['pool_rows'])+[dict(m,subset='bridge_positive',lineage_id=m['pair_id']) for m in e['members']]
    lookup={r['member_id']:r for r in rows};by_pair={r['pair_id']:r['member_id'] for r in e['members']}
    p={k:old[k] for k in ('names','initialization','evaluation','held_members')}
    p.update(pool_rows=rows,schedules={},brightness_factors={},training_config={},replacement_positions={},listings={},ledger={})
    paths=[pp,ep,Path(__file__).resolve()]
    for seed in (7,17,27):
        a=old['schedules'][f'VM-{seed}'];b=list(a);positions=[]
        for i,mid in enumerate(a):
            row=lookup[mid]
            if 'full_truth' in row and row.get('variant') in ('warm','cool'):
                b[i]=by_pair[row['pair_id']];positions.append(i)
        p['replacement_positions'][str(seed)]=positions
        for arm,seq in (('R',a),('G',b)):
            key=f'{arm}-{seed}';p['schedules'][key]=seq
            p['brightness_factors'][key]=old['brightness_factors'][f'VM-{seed}'];p['training_config'][key]=old['training_config'][f'VM-{seed}']
            listing=OUT/(key+'.txt');text=''.join(lookup[m]['image_path']+'\n' for m in sorted(set(seq)))
            if listing.exists() and listing.read_text()!=text:raise ValueError('Listing drift')
            if not listing.exists():listing.write_text(text)
            p['listings'][key]=str(listing);paths.append(listing)
            p['ledger'][key]=[dict(first_step=start//6+1,members=dict(Counter(seq[start:start+300])),
                instances={c:sum(lookup[m]['class_instances'].get(c,0) for m in seq[start:start+300]) for c in p['names']},
                lineages=dict(Counter(lookup[m]['lineage_id'] for m in seq[start:start+300]))) for start in range(0,2700,300)]
    checks(p,old)
    paths += [Path(r[k+'_path']) for r in rows for k in ('image','label')]
    return prior.frozen(dest,dict(**p,status='real_file_sequences_frozen_loader_pending',training_ready=False,training_started=False,
        training_admitted=False,promotable=False,reference='Historical low-dose VM, all three seeds; R loader reproduces it.',
        interpretation='Replace 30 warm/cool image exposures with same-source gray target bodies. No additional exposures.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(freeze()['status'])
