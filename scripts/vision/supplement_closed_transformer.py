"""One authorized four-condition supplement. Default freezes; never trains."""
import argparse
import asyncio
import copy
import fcntl
import shutil
from pathlib import Path
from scripts.vision import closed_exterior_material_capture as base
from scripts.vision.design_full_scene_poses import screen
from src.vision.canonical.plan import pose_close, write_record, read_record
from src.vision.canonical.gates import validate_preflight
from src.vision.canonical.collect import collect

prior=base.prior
OUT=base.OUT/'transformer-supplement-v1'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest);prior.verify(p);return p
    p=base.freeze();layout=next(x for x in p['layouts'] if x['layout_id']=='layout-B')
    source=base.OUT/'plans/layout-B/original/plan.json';plan=read_record(source)
    previous=layout['selected']+next(x for x in base.old.freeze()['layouts'] if x['layout_id']=='layout-B')['selected']
    ranked=[]
    for row in layout['geometric_candidates']:
        v=row['view']
        if v['category']!='transformer' or any(pose_close(v,x) for x in previous):continue
        boxes=screen(v,plan['objects']);overlap=0.
        for name,a in boxes.items():
            area=(a[2]-a[0])*(a[3]-a[1])
            for other,b in boxes.items():
                if other!=name:overlap=max(overlap,max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))/area)
        rank=(overlap,len(boxes),abs(v['height']-5),v['distance'],v['object_id'],v['bearing'],v['view_id'])
        ranked.append(dict(view=v,rank=rank,projected_boxes=boxes))
    if not ranked:raise ValueError('No bounded unused transformer pose')
    ranked.sort(key=lambda r:r['rank']);view=copy.deepcopy(ranked[0]['view'])
    view['pair_id']='layout-B:closed:transformer:supplement-1'
    OUT.mkdir(parents=True,exist_ok=False);runs=[]
    deps=[base.OUT/'protocol.json',base.OUT/'visual-review.json',base.OUT/'original-replays-repaired-v1/verified-progress.json',Path(__file__)]
    for variant in base.old.PALETTES:
        ancestor=base.OUT/'plans/layout-B'/variant;original=read_record(ancestor/'plan.json')
        folder=OUT/'plans'/variant;folder.mkdir(parents=True)
        for name,digest in original['files'].items():
            src=ancestor/name
            if prior.file_sha256(src)!=digest:raise ValueError('Ancestor changed')
            shutil.copy2(src,folder/name);deps.append(src)
        derived={k:copy.deepcopy(v) for k,v in original.items() if k!='identity'}
        derived.update(calibration_views=[view],pilot_views=[],diagnostic_purpose='authorized_one_pose_transformer_supplement')
        write_record(folder/'plan.json',derived);validate_preflight(derived,folder,[view]);deps.extend(folder.iterdir())
        runs.append(dict(variant=variant,plan_path=str(folder/'plan.json')))
    return prior.frozen(dest,dict(status='one_pose_four_conditions_frozen',selected=view,candidates=ranked,runs=runs,
        authorization='2026-09-14 user 确认: one additional layout-B transformer pose, four candidates beyond 64; then quality gates and training start',
        selection_rule='Minimum maximum projected target-box overlap, then target count, height distance from 5m, distance, device ID, bearing, view ID. Geometry is only a prefilter, not visibility evidence.',
        training_started=False,training_ready=False,training_admitted=False,promotable=False,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


async def run():
    p=freeze();dest=OUT/'capture-completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rows=[];deps=[OUT/'protocol.json']
        for run in p['runs']:
            folder=OUT/'captures'/run['variant'];rp=folder/'collection-receipt.json'
            if rp.exists():r=read_record(rp)
            elif folder.exists():raise ValueError('Incomplete attempt preserved')
            else:r=await collect(run['plan_path'],folder,mode='calibration')
            deps.append(rp);ok=r['status']=='complete_pending_review' and len(r['views'])==1 and r['views'][0]['status']=='captured'
            rows.append(dict(variant=run['variant'],receipt=str(rp),captured=ok));print(rows[-1],flush=True)
            if not ok:break
        prior.frozen(dest,dict(status='four_captured_review_required' if len(rows)==4 and all(x['captured'] for x in rows) else 'capture_blocked',runs=rows,
            training_started=False,training_ready=False,training_admitted=False,promotable=False,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:print(freeze()['selected'],'NO_CAPTURE_NO_TRAINING')
