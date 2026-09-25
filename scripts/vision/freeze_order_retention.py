"""Freeze reviewed batch permutations and independently exported training inputs."""
import shutil
from collections import Counter
from pathlib import Path
import yaml
from scripts.vision.exposure_order_retention import *
from scripts.vision.review_exposure_order_retention import validate_decisions
from scripts.vision.order_retention_runtime import overrides

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():verify_tree(dest);return read(dest)
    verify_tree(PRIOR/'numerical-audit-v1.json');verify_tree(OUT/'review.json')
    e=read(OUT/'evidence.json');review=read(OUT/'review.json');validate_decisions(e,review['decisions'])
    p=read(PRIOR/'protocol.json');lookup={r['member_id']:r for r in p['pool_rows']}
    from scripts.vision.design_visibility_repair_contrast import SOURCE as VIS
    tp=VIS.parent/'trace.json';trace=read(tp)
    allowed=('original','neutral_bridge','background_bridge')
    vm={m:'common' for m,r in lookup.items() if r['subset']!='bridge_positive'}
    for f in trace['frames']:
        if f['member_id'] in lookup:
            if f['variant'] not in allowed or f['member_id'] in vm:raise ValueError('Ambiguous variant identity')
            vm[f['member_id']]=f['variant']
    schedules={};batches_by_seed={};batch_orders={};ledgers={}
    for seed in (7,17,27):
        seq=p['schedules'][f'retained_appearance-450-{seed}'];batches,order=permutation(seq,vm,seed)
        batches_by_seed[str(seed)]=[dict(batch_id=f'{seed}:{i:03}:{object_sha256(b)[:16]}',members=b) for i,b in enumerate(batches)]
        for arm,ids in [('staged',list(range(450))),('interleaved',order)]:
            key=f'{arm}-450-{seed}';draws=[m for i in ids for m in batches[i]]
            if Counter(draws)!=Counter(seq):raise ValueError('Member exposure changed')
            schedules[key]=draws;batch_orders[key]=ids
            ledgers[key]=dict(total=exposures(p['pool_rows'],draws),windows_50_steps=[exposures(p['pool_rows'],draws[i:i+300]) for i in range(0,2700,300)])
        if ledgers[f'staged-450-{seed}']['total']!=ledgers[f'interleaved-450-{seed}']['total']:raise ValueError('Supervision/lineage exposure changed')
    paths=[PRIOR/'protocol.json',PRIOR/'numerical-audit-v1.json',OUT/'review.json',tp,Path(__file__),
        ROOT/'scripts/vision/exposure_order_retention.py',ROOT/'scripts/vision/order_retention_runtime.py',ROOT/'scripts/vision/preflight_order_retention.py',ROOT/'scripts/vision/train_order_retention.py']
    for seed in (7,17,27):paths.append(PRIOR/f'evaluation-retained_reference-450-{seed}.json')
    hp=Path(p['evaluation']['historical_reference']);paths.append(hp)
    inputs={str(path):file_sha256(path) for path in paths}
    export=OUT/'export-attempt-001'
    if export.exists():raise ValueError('Incomplete export preserved; new version required')
    (export/'images').mkdir(parents=True);(export/'labels').mkdir()
    rows=[]
    for i,mid in enumerate(sorted(set(m for seq in schedules.values() for m in seq))):
        old=lookup[mid];row={**old,'source_image_path':old['image_path'],'source_label_path':old['label_path']}
        for kind,folder in [('image','images'),('label','labels')]:
            src=Path(old[kind+'_path'])
            if file_sha256(src)!=old[kind+'_sha256']:raise ValueError('Changed training member')
            target=export/folder/(f'{i:03}'+src.suffix);shutil.copy2(src,target)
            if file_sha256(target)!=old[kind+'_sha256']:raise ValueError('Export mismatch')
            row[kind+'_path']=str(target);inputs[str(target)]=old[kind+'_sha256']
        rows.append(row)
    lookup={r['member_id']:r for r in rows};datasets={};listings={}
    for key in KEYS:
        listing=export/f'{key}.txt';listing.write_text('\n'.join(lookup[mid]['image_path'] for mid in sorted(set(schedules[key])))+'\n')
        config=export/f'{key}.yaml';config.write_text(yaml.safe_dump(dict(path=str(export),train=str(listing),val=str(listing),names=list(NAMES))))
        datasets[key]=str(config);listings[key]=str(listing)
        for path in (listing,config):inputs[str(path)]=file_sha256(path)
    return frozen(dest,dict(status='frozen_preflight_pending',schedules=schedules,batches=batches_by_seed,batch_orders=batch_orders,
        variant_by_member=vm,ledger=ledgers,exposures={k:v['total'] for k,v in ledgers.items()},pool_rows=rows,datasets=datasets,listings=listings,names=list(NAMES),
        controls=p['controls'],runtime_overrides={str(s):overrides(s) for s in (7,17,27)},initialization=p['initialization'],evaluation=p['evaluation'],
        acceptance_policy=p['acceptance_policy'],retention={**p['retention'],'same_budget_reference':'retained_reference-450',
            'comparison_arm':'staged is order comparator, not ability reference'},
        candidate_policy='No automatic candidate; all seeds and formal gates plus explicit review required; no promotion',
        inputs=inputs))

if __name__=='__main__':print('FROZEN',freeze()['identity'],flush=True)
