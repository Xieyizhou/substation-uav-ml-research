"""Deterministic fixed-budget color retention and gray-scope contrasts."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from scripts.vision.prepare_material_retention_coverage import OUT,prior,CAND
from scripts.vision import train_compensated_material as source

VERSION='material-retention-coverage-v1'
COLORS=('warm','cool','gray')


def allocate(positions,seed):
    groups=defaultdict(list)
    for x in positions:groups[x['planned_class']].append(x)
    if len(groups)!=4:raise ValueError('Missing planned class')
    result={}
    for category,rows in sorted(groups.items()):
        if len(rows)%3:raise ValueError('Cannot split class exposures into exact thirds')
        ordered=sorted(rows,key=lambda x:sha256(f"{VERSION}|{seed}|{category}|{x['source_id']}|{x['position']}".encode()).hexdigest())
        for i,row in enumerate(ordered):result[row['position']]=COLORS[i%3]
    if Counter(result.values())!=Counter(warm=10,cool=10,gray=10):raise ValueError('Color budget mismatch')
    return result


def checks(p,old):
    rows={r['member_id']:r for r in p['pool_rows']}
    for seed in (7,17,27):
        ref=old['schedules'][f'VM-{seed}'];positions=p['treatment_positions'][str(seed)]
        allocated=allocate(positions,seed)
        if allocated!={int(k):v for k,v in p['allocation'][str(seed)].items()}:raise ValueError('Allocation drift')
        affected={r['position'] for r in positions}
        for arm in ('R','T','A'):
            key=f'{arm}-{seed}';seq=p['schedules'][key]
            if len(seq)!=2700 or (arm=='R' and seq!=ref):raise ValueError('Budget/reference drift')
            if p['brightness_factors'][key]!=old['brightness_factors'][f'VM-{seed}'] or p['training_config'][key]!=old['training_config'][f'VM-{seed}']:raise ValueError('Runtime configuration drift')
            for i,(a,b) in enumerate(zip(ref,seq,strict=True)):
                if i not in affected and a!=b:raise ValueError('Untreated/negative position changed')
                if a!=b:
                    if rows[a]['pair_id']!=rows[b]['pair_id'] or rows[a]['class_instances']!=rows[b]['class_instances']:raise ValueError('Source or class exposure changed')
                    if Path(rows[a]['label_path']).read_bytes()!=Path(rows[b]['label_path']).read_bytes():raise ValueError('Complete labels changed')
            if set(seq)&set(p['held_members']):raise ValueError('Held member restored')
        for x in positions:
            i=x['position'];color=allocated[i]
            t=rows[p['schedules'][f'T-{seed}'][i]];a=rows[p['schedules'][f'A-{seed}'][i]]
            if color=='gray':
                if t['variant']!='gray_target_body' or a['variant']!='gray_all_body':raise ValueError('Gray scope changed')
            elif t['member_id']!=a['member_id'] or t['variant']!=color:raise ValueError('Color retention changed')
    for row in rows.values():
        for k in ('image','label'):
            if prior.file_sha256(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('Stale member')


def freeze():
    pp=source.OUT/'protocol.json';old=prior.read(pp);prior.verify(old)
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);checks(p,old);return p
    ep=OUT/'candidate-export-v1/manifest.json';e=prior.read(ep);prior.verify(e)
    cp=CAND/'reviewed-completion.json';c=prior.read(cp);prior.verify(c)
    original={r['member_id']:r for r in c['members']}
    from scripts.vision.resolve_material_scope_duplicates import main as resolve,inspect
    resolution=resolve()
    if len(e['members'])!=24 or resolution['groups']!=inspect(e):raise ValueError('Candidate gate')
    rows=list(old['pool_rows'])+[dict(r,subset='bridge_positive',lineage_id=r['pair_id']) for r in e['members']]
    lookup={r['member_id']:r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate member')
    variants={(r['pair_id'],r['variant']):r['member_id'] for r in rows if 'full_truth' in r and r.get('variant') in ('warm','cool','gray_target_body','gray_all_body')}
    p={k:old[k] for k in ('names','initialization','evaluation','held_members')}
    p.update(pool_rows=rows,schedules={},brightness_factors={},training_config={},listings={},ledger={},treatment_positions={},allocation={})
    paths=[pp,ep,cp,OUT/'duplicate-resolution.json',Path(__file__).resolve()]
    for seed in (7,17,27):
        ref=old['schedules'][f'VM-{seed}'];positions=[]
        for i,mid in enumerate(ref):
            row=lookup[mid]
            if 'full_truth' not in row or row.get('variant') not in ('warm','cool'):continue
            src=original[mid]
            positions.append(dict(position=i,source_id=src['source_pose_id'],planned_class=src['class_name'] if 'class_name' in src else src.get('planned_class'),pair_id=row['pair_id']))
            if positions[-1]['planned_class'] is None:
                from scripts.vision.audit_material_transfer_scope import resolve_truth
                ts=resolve_truth(src['full_truth'],src['instance_mapping'])
                positions[-1]['planned_class']=next(t['category'] for t in ts if t['object_id']==src['planned_object_id'])
        if len(positions)!=30:raise ValueError('Expected thirty material slots')
        colors=allocate(positions,seed);p['treatment_positions'][str(seed)]=positions;p['allocation'][str(seed)]={str(k):v for k,v in colors.items()}
        for arm in ('R','T','A'):
            key=f'{arm}-{seed}';seq=list(ref)
            if arm!='R':
                for x in positions:
                    color=colors[x['position']];variant=('gray_target_body' if arm=='T' else 'gray_all_body') if color=='gray' else color
                    seq[x['position']]=variants[(x['pair_id'],variant)]
            p['schedules'][key]=seq;p['brightness_factors'][key]=old['brightness_factors'][f'VM-{seed}'];p['training_config'][key]=old['training_config'][f'VM-{seed}']
            listing=OUT/(key+'.txt');listing.write_text(''.join(lookup[mid]['image_path']+'\n' for mid in sorted(set(seq))))
            p['listings'][key]=str(listing);paths.append(listing)
            p['ledger'][key]=[dict(first_step=start//6+1,members=dict(Counter(seq[start:start+300])),instances={cl:sum(lookup[mid]['class_instances'].get(cl,0) for mid in seq[start:start+300]) for cl in p['names']},lineages=dict(Counter(lookup[mid]['lineage_id'] for mid in seq[start:start+300]))) for start in range(0,2700,300)]
    checks(p,old)
    paths.extend(Path(r[k+'_path']) for r in rows for k in ('image','label'))
    return prior.frozen(dest,dict(**p,status='color_and_scope_sequences_frozen_loader_pending',training_ready=False,training_started=False,
        contrasts=['T versus historical G: retain warm/cool at fixed 30-image budget','A versus T: only gray coverage scope changes at the same ten positions'],
        limits=['No factorial interaction estimate; no all-gray all-body training arm.','Not every source appears in all three colors within each seed.','Shared poses/assets; class image quota is not full-instance condition balance.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(freeze()['status'])
