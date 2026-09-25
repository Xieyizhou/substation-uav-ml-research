"""Evidence and bounded exact replays for the authorized supplement; no approval."""
import argparse
import asyncio
import shutil
from pathlib import Path
from unittest.mock import patch
from scripts.vision import supplement_closed_transformer as s
from scripts.vision import build_closed_exterior_evidence as b
from scripts.vision.verify_designed_full_scene_poses import replay_frame,DESIGN

OUT=s.OUT;prior=s.prior


def protocol():
    p=s.freeze();v=p['selected'];runs=[dict(x,key=x['variant']) for x in p['runs']]
    units=[dict(key=x['variant'],run_key=x['variant'],variant=x['variant'],plan_path=x['plan_path'],
        view_id=v['view_id'],pair_id=v['pair_id'],category=v['category'],object_id=v['object_id']) for x in runs]
    return dict(p,runs=runs,units=units)


def verified(run):
    p=protocol();u=next(x for x in p['units'] if x['variant']==run['variant'])
    cp=OUT/'captures'/u['variant']/'collection-receipt.json'
    row=s.base.old.verify_capture(u,cp);r=s.read_record(cp)
    return {row['view_id']:row},r['collection_checks']['instance_mapping'],cp


def evidence():
    p=protocol();done=prior.read(OUT/'capture-completion.json');prior.verify(done)
    if done['status']!='four_captured_review_required':raise ValueError('Capture incomplete')
    with patch.object(b,'OUT',OUT),patch.object(b,'freeze',protocol),patch.object(b,'verified_run',verified):b.build()
    return p


async def replay():
    p=evidence();dest=OUT/'replays';dest.mkdir(exist_ok=True);completion=dest/'completion.json'
    if completion.exists():prior.verify(prior.read(completion));return
    results=[];deps=[OUT/'protocol.json',Path(__file__)]
    for u in p['units']:
        rows,mapping,cp=verified(u);row=rows[u['view_id']];ep=OUT/'evidence'/u['key']/'evidence.json'
        e=prior.read(ep);prior.verify(e);unit=dest/u['variant'];unit.mkdir(exist_ok=True)
        sp=unit/'source.json'
        if not sp.exists():prior.frozen(sp,dict(row,inputs={str(cp):prior.file_sha256(cp)}))
        f=dict(member_id=u['key'],lineage_id=u['pair_id'],class_name=u['category'],review_ids=['frame'],
            source_image=row['rgb_path'],source_receipt=str(sp),source_plan=u['plan_path'],
            source_world=str(Path(u['plan_path']).parent/'world.sdf'),actual_pose=row['actual_pose'],
            world_name=s.read_record(u['plan_path'])['world_name'],instance_mapping=mapping,
            events=[dict(review_id='label-'+x['runtime_label'],runtime_label=int(x['runtime_label']),object_id=x['object_id'],bbox_xyxy=x['truth']['bbox_xyxy']) for x in e['events']])
        helper=unit/'gz_visibility_capture_cleanup_fixed'
        if not helper.exists():shutil.copy2(DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed',helper)
        pp=unit/'protocol.json'
        if not pp.exists():prior.frozen(pp,dict(frame=f,training_ready=False,inputs={str(x):prior.file_sha256(x) for x in (cp,ep,sp,helper,Path(__file__))}))
        prior.verify(prior.read(pp));rp,r=await replay_frame(f,unit);deps.extend([pp,rp,ep])
        results.append(dict(variant=u['variant'],receipt=str(rp),status=r['status'] if r else 'attempts_exhausted'))
        if not r or r['status']!='original_pixel_evidence_certified':break
    prior.frozen(completion,dict(status='four_exact_replays_review_required' if len(results)==4 and all(x['status']=='original_pixel_evidence_certified' for x in results) else 'replay_blocked',
        results=results,training_ready=False,training_started=False,training_admitted=False,promotable=False,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(replay())
    else:evidence();print('EVIDENCE_ONLY_NO_TRAINING')
