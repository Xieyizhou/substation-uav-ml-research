"""Four preselected sources: exact replay plus full-mask coverage before variants."""
import argparse,asyncio,fcntl,json,os,re,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.establish_material_view_candidates import OUT as SOURCE,HISTORY,prior
from scripts.vision.build_material_view_world_drafts import valid_clock
from scripts.vision.audit_material_mask_coverage import coverage
from scripts.vision import run_visibility_cleanup_validation as replay
from src.vision.canonical.plan import pose_close

OUT=SOURCE/'fresh-source-probe-v1'


def freeze():
    paths=[SOURCE/'source-inventory.json',SOURCE/'source-review.json',SOURCE/'N05-expansion-v1/reviewed-completion.json',HISTORY/'coverage-census.json']
    records=[prior.read(p) for p in paths]
    for r in records:prior.verify(r)
    OUT.mkdir(exist_ok=True);dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    inventory,old,holds,census=records
    excluded={s['lineage_id'] for s in old['sources']}|set(holds['denied_lineages'])
    seen=[m['actual_pose'] for m in census['members'] if m.get('actual_pose') and any(m['actual_exposures'].values())]
    groups={}
    for r in inventory['records']:
        if r['status']!='source_trace_passed_review_required' or r.get('lighting_id')!='light_normal' or r['lineage_id'] in excluded:continue
        if any(pose_close(r['actual_pose'],p) for p in seen):continue
        groups.setdefault(r['lineage_id'],[]).append(r)
    representatives=[]
    for rows in groups.values():
        s=min(rows,key=lambda r:({'material_original':0,'mat_cool_gray':1,'mat_warm_oxide':2,'mat_desaturated_green':3}.get(r['material_id'],9),r['source_member_id']))
        plan=prior.read(s['source_plan']);v=next(v for k in ('calibration_views','pilot_views') for v in plan.get(k,[]) if v['view_id']==s['source_view_id'])
        representatives.append(dict(s,distance=v['distance'],bearing=v['bearing']))
    selected=[];frames=[]
    for n,category in enumerate(('capacitor_bank','switchgear','reactor','transformer'),1):
        candidates=sorted([s for s in representatives if s['category']==category],key=lambda s:(s['distance'],s['object_id'],s['bearing'],s['source_view_id']))
        if not candidates:raise ValueError('No unheld source for '+category)
        s=dict(candidates[0],probe_id=f'P{n:02}');selected.append(s);sid=s['probe_id']
        sp=OUT/f'{sid}-source.json';prior.frozen(sp,dict(s['source_record'],inputs={s['source_capture']:prior.file_sha256(s['source_capture'])}))
        events=[]
        for j,t in enumerate(s['source_record']['truth']['objects']):
            label=int(re.search(r'instance-(\d+)-',t['annotation_id'])[1]);identity=s['instance_mapping'][str(label)]
            if identity['category']!=t['class_name']:raise ValueError('Instance class conflict')
            events.append(dict(review_id=f'{sid}-{j:02}',runtime_label=label,object_id=identity['object_id'],bbox_xyxy=t['bbox_xyxy']))
        frames.append(dict(member_id=s['source_member_id'],lineage_id=s['lineage_id'],class_name=category,review_ids=[sid],source_image=s['source_image'],source_receipt=str(sp),source_plan=s['source_plan'],source_world=s['source_world'],actual_pose=s['actual_pose'],world_name=s['world_name'],instance_mapping=s['instance_mapping'],events=events))
        paths += [sp,*[Path(s[k]) for k in ('source_image','source_capture','source_plan','source_world')]]
    helper=SOURCE/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed';shutil.copy2(helper,OUT/helper.name)
    paths += [helper,OUT/helper.name,Path(__file__),Path(replay.__file__),prior.ROOT/'scripts/vision/audit_material_mask_coverage.py',prior.ROOT/'scripts/vision/build_material_view_world_drafts.py']
    return prior.frozen(dest,dict(status='four_sources_frozen_before_replay',sources=selected,frames=frames,
        selection_rule='Exclude all prior twelve source groups and new holds; exclude resolved exposed poses; normal light, original preferred per group; first distance/object/bearing/view_id per class. Fixed four, no adaptive replacements or model-score selection.',
        limitations=['Shared complex layout/assets','Unresolved historical poses limit isolation','Saved material may not be original appearance; this is same-frame source replay, not V-arm capture','Historical admission chain remains unverified; no training'],
        max_attempts_per_frame=3,training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


async def run():
    p=freeze();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=(replay.OUT,replay.analyze,replay.command);oldlog=os.environ.get('GZ_LOG_PATH');replay.OUT=OUT
        logs=OUT/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
        def analyze(folder,f,fence):
            r=saved[1](folder,f,fence);checks=[]
            for n in range(1,4):
                a=np.fromfile(folder/f'first-stable-window/frame-{n}-mask.bin',dtype='u1').reshape(1080,1920,3)
                c=coverage(a,[e['runtime_label'] for e in f['events']],f['instance_mapping'])
                c['visible_pixels_by_label']={str(k):v for k,v in c['visible_pixels_by_label'].items()};checks.append(c)
            r['full_mask_coverage']=checks
            if any(c['missing_targets'] for c in checks):r.update(status='semantic_blocked',reason='Visible mapped target lacks full box; no variant generation')
            return r
        async def command(*args,**kw):
            if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):return await saved[2](*args,**kw)
            for _ in range(6):
                raw=await saved[2](*args,**dict(kw,timeout=5))
                if valid_clock(json.loads(raw)):return raw
                await asyncio.sleep(.25)
            raise ValueError('Clock preflight failed')
        replay.analyze=analyze;replay.command=command;results=[];paths=[OUT/'protocol.json']
        try:
            for f in p['frames']:
                last=None
                for n in range(1,4):
                    rp=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}/receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif rp.parent.exists():continue
                    else:last=await replay.attempt(f,n)
                    paths.append(rp)
                    if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
                    if last['status']!='technical_failure' or any(e in last.get('reason','') for e in ('TypeError:','KeyError:','AttributeError:')):break
                results.append(dict(probe_id=f['review_ids'][0],status=last['status'] if last else 'attempts_exhausted',receipt=str(rp),reason=last.get('reason','') if last else 'Incomplete retained'))
        finally:
            replay.OUT,replay.analyze,replay.command=saved
            if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
            else:os.environ['GZ_LOG_PATH']=oldlog
        prior.frozen(dest,dict(status='four_source_probes_complete_visual_review_required',units=results,training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
        print([(r['probe_id'],r['status']) for r in results])


def evidence():
    p=freeze();dest=OUT/'review';dest.mkdir(exist_ok=True);paths=[OUT/'protocol.json',Path(__file__)]
    for f in p['frames']:
        sid=f['review_ids'][0];im=Image.open(f['source_image']).convert('RGB');count=len(f['events'])
        canvas=Image.new('RGB',(1500,750+300*((count+2)//3)),'white');d=ImageDraw.Draw(canvas);over=im.copy();od=ImageDraw.Draw(over)
        for e in f['events']:od.rectangle(e['bbox_xyxy'],outline='red',width=3)
        over.thumbnail((1450,700));canvas.paste(over,(0,25));d.text((0,0),sid+' '+f['class_name'],fill='black')
        for j,e in enumerate(f['events']):
            crop=im.crop(tuple(e['bbox_xyxy']));op=dest/(e['review_id']+'.png');crop.save(op);paths.append(op);crop.thumbnail((480,270))
            x=j%3*500;y=750+j//3*300;d.text((x,y),e['review_id']+' '+e['object_id'],fill='black');canvas.paste(crop,(x,y+20))
        op=dest/(sid+'.png');canvas.save(op);paths.append(op)
    prior.frozen(dest/'evidence.json',dict(status='explicit_visual_review_required',training_ready=False,inputs={str(p):prior.file_sha256(p) for p in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');ap.add_argument('--evidence',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(run())
    elif a.evidence:evidence()
    else:freeze();print('FROZEN_NO_REPLAY_NO_TRAINING')
