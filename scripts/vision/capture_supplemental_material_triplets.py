"""Sixteen body-material variants from eight verified sources; no training."""
import argparse,asyncio,copy,fcntl,json,os,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.verify_supplemental_full_scene_poses import OUT as SOURCES,DESIGN,prior
from scripts.vision.build_material_view_world_drafts import valid_clock,material_nodes,check_only_materials,PALETTES
from scripts.vision.expand_material_view_n05 import mask_gate
from scripts.vision import run_visibility_cleanup_validation as replay

OUT=DESIGN/'body-material-triplets-v1'


def freeze():
    rp=SOURCES/'reviewed-completion.json';review=prior.read(rp);prior.verify(review)
    if review['status']!='eight_new_sources_verified_material_variants_not_started':raise ValueError('Sources not ready')
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    OUT.mkdir(exist_ok=True);worlds=OUT/'worlds';worlds.mkdir(exist_ok=True)
    helper=DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed'
    paths=[rp,helper,Path(__file__),Path(replay.__file__),prior.ROOT/'scripts/vision/build_material_view_world_drafts.py',prior.ROOT/'scripts/vision/expand_material_view_n05.py']
    selected={v['probe_id']:v for v in prior.read(DESIGN/'protocol.json')['selected']}
    rows=[]
    for source in review['sources']:
        sid=source['probe_id'];up=SOURCES/sid/'protocol.json';u=prior.read(up);prior.verify(u);frame=u['frame']
        control=SOURCES/sid/'replay'/sid/'attempt-01/receipt.json'
        r=prior.read(control);prior.verify(r)
        if r['status']!='original_pixel_evidence_certified' or any(x['missing_targets'] for x in r['full_mask_coverage']):raise ValueError('Source pixel/coverage gate failed')
        ref=control.parent/'first-stable-window/frame-1-mask.bin'
        base=ET.parse(frame['source_world']).getroot();name=selected[sid]['object_id']
        for variant,color in PALETTES.items():
            tree=copy.deepcopy(base);nodes=material_nodes(tree,{name},('body','reactor'))
            if len(nodes)!=2:raise ValueError('Expected unique planned equipment body')
            for n in nodes.values():n.text=color
            changes=check_only_materials(base,tree,{name},('body','reactor'))
            if len(changes)!=2:raise ValueError('Material did not change exactly two fields')
            wp=worlds/f'{sid}-{variant}.sdf';ET.ElementTree(tree).write(wp,encoding='utf-8',xml_declaration=True)
            rows.append(dict(probe_id=sid,variant=variant,key=f'{sid}/{variant}',frame=frame,world=str(wp),reference_mask=str(ref),target_object_id=name,changed_fields=changes))
            paths.append(wp)
        paths += [up,control,ref,Path(frame['source_image']),Path(frame['source_receipt']),Path(frame['source_world'])]
    return prior.frozen(dest,dict(status='sixteen_variants_frozen',units=rows,helper=str(helper),max_attempts=3,
        policy='Change only planned equipment body ambient/diffuse. No panel/base/geometry/light/camera changes. Exact per-pixel instance mask and full box membership retained, then fresh visual review. All three variants grouped, not independent samples.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


async def unit(frame,variant,world=None,reference=None):
    folder=OUT/variant;folder.mkdir(parents=True,exist_ok=True);helper=folder/'gz_visibility_capture_cleanup_fixed'
    if not helper.exists():shutil.copy2(prior.read(OUT/'protocol.json')['helper'],helper)
    pp=folder/'protocol.json'
    deps=[OUT/'protocol.json',helper]
    if world:deps.append(Path(world))
    if reference:deps.append(reference)
    if not pp.exists():prior.frozen(pp,dict(variant=variant,training_ready=False,inputs={str(p):prior.file_sha256(p) for p in deps}))
    prior.verify(prior.read(pp))
    saved=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.command);oldlog=os.environ.get('GZ_LOG_PATH')
    replay.OUT=folder;logs=folder/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
    if world:
        tree=ET.parse(world)
        replay.add_sensor=lambda original:saved[1](copy.deepcopy(tree))
        replay.validate_world=lambda original,derived:saved[2](tree,derived)
        def analyze(path,f,fence):
            result=saved[3](path,f,fence);mask_gate(result,path,reference)
            result.update(status='candidate_rendered_review_pending',reason='Expected RGB appearance change; exact source mask and complete boxes preserved. No admission.')
            return result
        replay.analyze=analyze
    async def command(*args,**kw):
        if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):return await saved[4](*args,**kw)
        for _ in range(6):
            raw=await saved[4](*args,**dict(kw,timeout=5))
            if valid_clock(json.loads(raw)):return raw
            await asyncio.sleep(.25)
        raise ValueError('Clock preflight failed')
    replay.command=command;last=None;rp=None
    try:
        for n in range(1,4):
            rp=folder/f'replay/{frame["review_ids"][0]}/attempt-{n:02}/receipt.json'
            if rp.exists():last=prior.read(rp);prior.verify(last)
            elif rp.parent.exists():continue
            else:last=await replay.attempt(frame,n)
            if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
            if last['status']!='technical_failure' or any(e in last.get('reason','') for e in ('TypeError:','KeyError:','AttributeError:')):break
    finally:
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.command=saved
        if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
        else:os.environ['GZ_LOG_PATH']=oldlog
    return dict(variant=variant,status=last['status'] if last else 'attempts_exhausted',receipt=str(rp))

async def run():
    p=freeze();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[];blocked=set();paths=[OUT/'protocol.json']
        for row in p['units']:
            if row['probe_id'] in blocked:continue
            result=await unit(row['frame'],row['key'],row['world'],Path(row['reference_mask']))
            result['probe_id']=row['probe_id'];result['appearance']=row['variant'];results.append(result)
            paths.append(Path(result['receipt']))
            if result['status']!='candidate_rendered_review_pending':blocked.add(row['probe_id'])
        prior.frozen(dest,dict(status='sixteen_variants_captured_review_pending' if len(results)==16 and not blocked else 'incomplete_triplets_held',units=results,
            training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
        print([(r['variant'],r['status']) for r in results])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:freeze();print('FROZEN_NO_CAPTURE_NO_TRAINING')
