"""Freeze paired data and exact schedules; no training entry."""
import copy,hashlib,shutil
from collections import Counter
from pathlib import Path
from scripts.vision.unified_hold_lighting_counts import OUT as DESIGN,prior,SOURCE
from scripts.vision import lineage_capped_control as sequencing
from scripts.vision.exposure_protocol import exposures
from scripts.vision.record_unified_lighting_review import validate
from scripts.vision.finalize_l05_hold_block import validate_counts
from scripts.vision.brightness_lr_retention import overrides

OUT=DESIGN/'training-preflight'
VERSION='unified-hold-physical-lighting-control-v1'

def paired_sequences(rows,original,counts,quotas,variants,seed):
    old=sequencing.digest
    sequencing.digest=lambda *parts:hashlib.sha256(':'.join(map(str,(VERSION,*parts))).encode()).hexdigest()
    try:ref=sequencing.sequence(rows,original,counts,seed)
    finally:sequencing.digest=old
    light=list(ref);positions={}
    for member,q in sorted(quotas.items()):
        choices=[i for i,m in enumerate(ref) if m==member]
        if q<=0 or 2*q>len(choices):raise ValueError('Invalid lighting quota')
        selected=sorted(choices,key=lambda i:(hashlib.sha256(f'{VERSION}:light:{seed}:{member}:{i}'.encode()).hexdigest(),i))[:q]
        for i in selected:light[i]=variants[member];positions[str(i)]=member
    if sum(a!=b for a,b in zip(ref,light))!=sum(quotas.values()):raise ValueError('Replacement mismatch')
    return ref,light,dict(sorted(positions.items(),key=lambda x:int(x[0])))

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    dp,cp=DESIGN/'design.json',DESIGN/'counts.json';ep,rp=DESIGN/'light-review/evidence.json',DESIGN/'light-review/review.json'
    paths=[dp,cp,ep,rp,SOURCE/'protocol.json',Path(__file__).resolve(),Path(sequencing.__file__)]
    for x in paths[:5]:prior.verify(prior.read(x))
    d,c,e,r,previous=map(prior.read,paths[:5]);validate(e,r['decisions'])
    if c['status']!='counts_verified_capture_pending':raise ValueError('Count gate not passed')
    if prior.file_sha256(Path(d['initialization']['path']))!=d['initialization']['sha256']:raise ValueError('Initialization changed')
    paths.append(Path(d['initialization']['path']))
    OUT.mkdir(exist_ok=True);images=OUT/'export/images';labels=OUT/'export/labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(exist_ok=True)
    rows=copy.deepcopy(d['pool_rows']);idx={x['member_id']:x for x in rows};active={m for result in c['results'].values() for m,v in result['counts'].items() if v}
    for i,row in enumerate(rows):
        row['source_member_id']=row['member_id'];row['variant']='original'
        for kind in ('image','label'):
            original=Path(row[kind+'_path']);paths.append(original)
            if prior.file_sha256(original)!=row[kind+'_sha256']:raise ValueError('Original member changed')
            if row['member_id'] in active:
                target=(images if kind=='image' else labels)/(f'{i:03}'+original.suffix)
                if target.exists() and prior.file_sha256(target)!=row[kind+'_sha256']:raise ValueError('Existing export differs')
                if not target.exists():shutil.copy2(original,target)
                row[kind+'_path']=str(target);paths.append(target)
    variants={};by_pair={f['pair_id']:f for f in e['events']}
    for f in d['light_sources']:
        mid=f['member_id'];event=by_pair[f['id']];row=copy.deepcopy(idx[mid]);vid=mid+'|physical-lighting-v1';variants[mid]=vid
        ip=images/(f['id']+'-physical.png');lp=labels/(f['id']+'-physical.txt')
        for source,target in ((Path(event['image']),ip),(Path(row['label_path']),lp)):
            if target.exists() and prior.file_sha256(target)!=prior.file_sha256(source):raise ValueError('Variant export differs')
            if not target.exists():shutil.copy2(source,target)
            paths += [source,target]
        row.update(member_id=vid,source_member_id=mid,variant='physical-lighting',pair_id=f['id'],image_path=str(ip),label_path=str(lp),image_sha256=prior.file_sha256(ip),label_sha256=prior.file_sha256(lp))
        if row['label_sha256']!=idx[mid]['label_sha256']:raise ValueError('Variant changed full labels')
        rows.append(row)
    schedules={};factors={};listings={};ledger={};positions={};cfg={}
    for seed in (7,17,27):
        historical=SOURCE/'training'/f'brightness-450-{seed}'/'completion.json';receipt=prior.read(historical);prior.verify(receipt)
        xp=Path(receipt['exposure_path']);ex=prior.read(xp);prior.verify(ex);paths += [historical,xp]
        original=d['original_sequences'][f'brightness-450-{seed}']
        if ex['draws']!=original:raise ValueError('Reference actual/plan discrepancy')
        tc=c['results'][str(seed)]['counts'];validate_counts(d['pool_rows'],Counter(original),tc,set(d['held_members']),set(d['allowlist']),d['risk_groups'])
        ref,light,pos=paired_sequences(d['pool_rows'],original,tc,c['light_quotas'],variants,seed);positions[str(seed)]=pos
        for group,seq in [('R-clean',ref),('L-physical',light)]:
            key=f'{group}-{seed}';schedules[key]=seq;factors[key]=d['brightness_factors'][f'brightness-450-{seed}'];cfg[key]=overrides(seed)
            lookup={x['member_id']:x for x in rows};listing=OUT/(key+'.txt');content='\n'.join(lookup[m]['image_path'] for m in sorted(set(seq)))+'\n'
            if listing.exists() and listing.read_text()!=content:raise ValueError('Listing changed')
            if not listing.exists():listing.write_text(content)
            listings[key]=str(listing);paths.append(listing)
            ledger[key]=dict(total=exposures(rows,seq),windows_50_steps=[exposures(rows,seq[i:i+300]) for i in range(0,2700,300)])
        a,b=ledger[f'R-clean-{seed}']['total'],ledger[f'L-physical-{seed}']['total']
        for field in ('draws','by_subset','class_instance_exposure','by_lineage'):
            if a[field]!=b[field]:raise ValueError('Paired supervision/lineage differs')
    return prior.frozen(dest,dict(status='paired_sequences_frozen_loader_pending',pool_rows=rows,names=previous['names'],schedules=schedules,
        brightness_factors=factors,listings=listings,ledger=ledger,replacement_positions=positions,variants=variants,held_members=d['held_members'],
        initialization=d['initialization'],evaluation=d['evaluation'],acceptance_policy=d['acceptance_policy'],retention=d['retention'],
        direct_retention=dict(reference='R-clean same seed',tolerance=.05,conditions=['original','lighting'],metrics=['all_instance_recall','per_class_instance_recall']),
        environment=previous['environment'],training_config=cfg,training_ready=False,training_started=False,
        interpretation='Same revised source exposure and labels; 70 fixed positions use physical-lighting RGB. Four source poses, not new independent scenes.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':freeze();print('PAIRED_SEQUENCE_FROZEN_NO_TRAINING')
