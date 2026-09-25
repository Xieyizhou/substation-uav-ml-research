"""Independent manifests, fixed slot replacement, optimizer-free loader checks."""
import argparse
import fcntl
import hashlib
import shutil
import traceback
from collections import Counter
from pathlib import Path
from scripts.vision.design_supervision_hold_control import OUT as DESIGN, EVENTS
from scripts.vision.structure_fit import OUT as FIT, SOURCE, ROOT, read, verify, frozen, file_sha256, truth_for
from scripts.vision.supervision_risk_proposal import closure
from scripts.vision.resume_supervision_risk_pilot import OUT as PILOT
from scripts.vision.exposure_protocol import exposures
from scripts.vision.order_retention_runtime import preflight_cell, check_actual, overrides
from scripts.vision.check_structure_fit_sources import equal_rgb, label_correspondence
from scripts.vision.exposure_order_retention import baseline_verify

OUT=SOURCE/'whole-image-hold-control-v1'

def replace_slots(rows, original, held, seed):
    idx={r['member_id']:r for r in rows}
    if len(idx)!=len(rows) or len(original)!=2700 or set(original)-set(idx):raise ValueError('Invalid members or budget')
    result=list(original);offset=Counter();streams={}
    for i,mid in enumerate(original):
        if mid not in held:continue
        subset=idx[mid]['subset'];eligible=sorted(m for m,r in idx.items() if r['subset']==subset and m not in held)
        if not eligible:raise ValueError('Empty eligible subset')
        cycle=offset[subset]//len(eligible);position=offset[subset]%len(eligible)
        key=(subset,cycle)
        if key not in streams:
            streams[key]=sorted(eligible,key=lambda m:(hashlib.sha256(f'whole-image-hold-control-v1:{seed}:{subset}:{cycle}:{m}'.encode()).hexdigest(),m))
        result[i]=streams[key][position];offset[subset]+=1
    validate_slots(rows,original,result,held)
    return result

def validate_slots(rows,original,result,held):
    idx={r['member_id']:r for r in rows}
    if len(result)!=2700 or len(original)!=2700 or set(result)-set(idx) or set(result)&set(held):raise ValueError('Invalid replacement')
    for a,b in zip(original,result):
        if (a not in held and a!=b) or idx[a]['subset']!=idx[b]['subset']:raise ValueError('Unauthorized slot change')

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():verify(read(dest));return read(dest)
    paths=[DESIGN/'design.json',FIT/'protocol.json',FIT/'member-source-trace.json',FIT/'review.json',
           FIT/'completion.json',PILOT/'updated-proposal.json',PILOT/'completion.json',SOURCE/'protocol.json',Path(__file__),
           ROOT/'scripts/vision/order_retention_runtime.py',ROOT/'scripts/vision/check_structure_fit_sources.py']
    docs={str(x):read(x) for x in paths if x.suffix=='.json'}
    for d in docs.values():verify(d)
    p=docs[str(FIT/'protocol.json')];old=docs[str(SOURCE/'protocol.json')];src=docs[str(FIT/'member-source-trace.json')]
    held=set(read(DESIGN/'design.json')['held_member_ids'])
    if set(closure(p['rows'],held,src['rows']))!=held:raise ValueError('Registered closure expanded; revise design')
    if len(p['rows'])!=236 or len(held)!=4:raise ValueError('Scope changed')
    source={r['member_id']:r for r in src['rows']}
    if len(source)!=len(src['rows']) or set(source)!={r['member_id'] for r in p['rows']}:raise ValueError('Ambiguous source')
    inputs={str(x):file_sha256(x) for x in paths}
    # Validate complete source supervision; preserve documented historical gaps, do not certify them away.
    for r in p['rows']:
        s=source[r['member_id']]
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale member')
        equal_rgb(r['image_path'],s['source_image'])
        cp=Path(s['source_image']).parents[1]/'collection-receipt.json'
        views=[v for v in read(cp)['views'] if v['view_id']==s['source_view_id']]
        if len(views)!=1:raise ValueError('Ambiguous source view')
        label_correspondence(truth_for(r),views[0]['truth']['objects'])
        inputs[str(cp)]=file_sha256(cp)
    frozen(OUT/'manifest.json',dict(status='independent_development_manifest_not_training_admission',
        retained=[r for r in p['rows'] if r['member_id'] not in held],held=[r for r in p['rows'] if r['member_id'] in held],
        source_gaps=[dict(member_id=s['member_id'],gaps=s['gaps'],actual_annotation_mode=s['actual_annotation_mode']) for s in src['rows'] if s['gaps']],
        metadata_sidecar_events=['T04','T26','T32'],metadata_note='not_truncated describes converter clipping, not physical visibility',
        no_new_review_decisions=True,inputs=inputs))
    schedules={}
    for seed in (7,17,27):
        original=p['models'][f'interleaved-450-{seed}']['draws']
        schedules[f'reference-450-{seed}']=original
        schedules[f'hold-450-{seed}']=replace_slots(p['rows'],original,held,seed)
    export=OUT/'export';(export/'images').mkdir(parents=True);(export/'labels').mkdir()
    rows=[]
    for i,r in enumerate(p['rows']):
        row=dict(r)
        for kind,folder in [('image','images'),('label','labels')]:
            path=export/folder/(f'{i:03}'+Path(r[kind+'_path']).suffix);shutil.copy2(r[kind+'_path'],path)
            if file_sha256(path)!=r[kind+'_sha256']:raise ValueError('Copy mismatch')
            row[kind+'_path']=str(path);inputs[str(path)]=r[kind+'_sha256']
        rows.append(row)
    idx={r['member_id']:r for r in rows};listings={};ledgers={}
    for key,seq in schedules.items():
        path=export/(key+'.txt');path.write_text('\n'.join(idx[m]['image_path'] for m in sorted(set(seq)))+'\n')
        listings[key]=str(path);inputs[str(path)]=file_sha256(path)
        ledgers[key]=dict(total=exposures(rows,seq),windows_50_steps=[exposures(rows,seq[i:i+300]) for i in range(0,2700,300)])
    inputs[str(OUT/'manifest.json')]=file_sha256(OUT/'manifest.json')
    return frozen(dest,dict(status='frozen_preflight_pending',pool_rows=rows,schedules=schedules,listings=listings,ledger=ledgers,
        held_member_ids=sorted(held),names=old['names'],controls=old['controls'],initialization=old['initialization'],
        runtime_overrides={str(s):overrides(s) for s in (7,17,27)},evaluation=old['evaluation'],acceptance_policy=old['acceptance_policy'],
        retention={**old['retention'],'comparison_arm':'unchanged interleaved is data-policy comparator only'},
        historical_source_gaps_preserved=True,training_started=False,inputs=inputs))

def preflight():
    p=freeze();verify(p)
    for key in p['schedules']:
        root=OUT/'preflight'/key;root.mkdir(parents=True,exist_ok=True);done=root/'completion.json'
        if done.exists():
            unit=read(done);verify(unit)
            if unit['protocol_identity']!=p['identity']:raise ValueError('Stale receipt')
            check_actual(p,key,unit['actual']);continue
        attempts=list(root.glob('attempt-*'))
        if len(attempts)>=3:raise ValueError('Attempt budget exhausted')
        attempt=root/f'attempt-{len(attempts)+1:03}';attempt.mkdir()
        try:
            unit=preflight_cell(p,key)
            frozen(done,dict(**unit,protocol_identity=p['identity'],inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}))
        except BaseException:
            frozen(attempt/'failure.json',dict(error=traceback.format_exc(),workers=0,child_processes_started=0));raise
        print('LOADER_PASS',key,flush=True)
    print('SIX_LOADERS_COMPLETE_FINAL_VERIFICATION_PENDING',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generate',action='store_true');ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.generate or a.preflight:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'build.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.preflight:preflight()
            else:freeze();print('FROZEN_NO_TRAINING')
    else:print('PREFLIGHT_ONLY_EXPLICIT_GENERATE_REQUIRED_NO_TRAINING')
