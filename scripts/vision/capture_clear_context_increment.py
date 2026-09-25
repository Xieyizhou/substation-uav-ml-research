"""Explicit capture and evidence only; no semantic auto-approval."""
import argparse
import asyncio
import math
import re
import shutil
import os
from pathlib import Path
from PIL import Image,ImageDraw
from src.vision.canonical.collect import collect
from src.vision.canonical.gates import instance_mapping
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256,object_sha256
from scripts.vision.prepare_clear_context_increment import OUT,freeze,checked


async def capture(attempt=1):
    if attempt not in (1,2,3):raise ValueError('Technical attempt cap')
    p=freeze();root=OUT/('capture' if attempt==1 else f'capture-attempt-{attempt:03}')
    for number in range(1,attempt):
        previous=OUT/('capture' if number==1 else f'capture-attempt-{number:03}')/'collection-receipt.json'
        prior=checked(previous)
        if prior['status']!='blocked' or prior['views']:raise ValueError('Only terminal startup failure with zero frames may retry here')
    if root.exists():
        cp=root/'collection-receipt.json'
        if not cp.exists():raise ValueError('Incomplete capture attempt retained; inspect before a separate technical attempt')
        r=checked(cp)
        if r['status']!='complete_pending_review':raise ValueError('Capture blocked; do not retry semantic failures')
        return r
    compiler=Path(shutil.which('clang++')).resolve()
    toolchain=OUT/f'toolchain-attempt-{attempt:03}.json'
    from src.sensors.gazebo_rgbd_native import SOURCE as bridge_source,BINARY as bridge_binary
    if not toolchain.exists():write_record(toolchain,dict(compiler=str(compiler),compiler_sha256=file_sha256(compiler),
        sdkroot_environment=os.environ.get('SDKROOT'),bridge_source=str(bridge_source),bridge_source_sha256=file_sha256(bridge_source),
        purpose='Record existing explicitly selected toolchain; no license agreement acceptance or simulator/world change',
        training_admitted=False,promotable=False,inputs={str(x):file_sha256(x) for x in (compiler,bridge_source,Path(__file__).resolve())}))
    result=await collect(p['plan_path'],root,mode='calibration')
    paths=[toolchain,root/'collection-receipt.json']
    if result['status']=='complete_pending_review':paths.append(bridge_binary)
    write_record(OUT/f'capture-run-{attempt:03}.json',dict(status=result['status'],collection_receipt=str((root/'collection-receipt.json').resolve()),
        bridge_binary_sha256=file_sha256(bridge_binary) if result['status']=='complete_pending_review' else None,
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths}))
    return result


def evidence():
    p=freeze();plan=checked(p['plan_path'])
    receipts=[x for x in OUT.glob('capture*/collection-receipt.json') if checked(x)['status']=='complete_pending_review']
    if len(receipts)!=1:raise ValueError('Expected one complete capture attempt')
    receipt=receipts[0];r=checked(receipt)
    if r['status']!='complete_pending_review' or r['plan_identity']!=plan['identity']:raise ValueError('Incomplete/mismatched capture')
    by={v['view_id']:v for v in plan['calibration_views']};mapping=instance_mapping(plan)
    if len(r['views'])!=12 or set(by)!={v['view_id'] for v in r['views']}:raise ValueError('Capture population mismatch')
    dest=OUT/'evidence';dest.mkdir(exist_ok=True);path=dest/'manifest.json'
    if path.exists():return checked(path)
    deps=[receipt,Path(p['plan_path']),Path(__file__)];frames=[]
    for run in OUT.glob('capture-run-*.json'):
        record=checked(run)
        if record['status']=='complete_pending_review':deps.append(run)
    for v in r['views']:
        view=by[v['view_id']];rid=view['review_id'];image=Path(v['rgb_path'])
        if v['status']!='captured' or file_sha256(image)!=v['image_sha256']:raise ValueError('Invalid captured RGB')
        if not v['target_checks']['planned_instance_present']:raise ValueError('Missing planned instance')
        rgb=Image.open(image).convert('RGB');truth=v['truth']['objects'];annotated=rgb.copy();draw=ImageDraw.Draw(annotated);objects=[]
        for j,o in enumerate(truth):
            label=int(re.search(r'instance-(\d+)',o['annotation_id'])[1]);target=mapping[label]
            if target['category']!=o['class_name']:raise ValueError('Instance category conflict')
            box=o['bbox_xyxy'];draw.rectangle(box,outline='#ff3333',width=4);draw.text((box[0],max(0,box[1]-16)),f'{j} {target["object_id"]}',fill='red')
            crop=dest/f'{rid}-box-{j:02}.png';rgb.crop(tuple(box)).save(crop)
            objects.append(dict(annotation_id=o['annotation_id'],scene_device_id=target['object_id'],class_name=o['class_name'],bbox_xyxy=box,crop=str(crop.resolve()),crop_sha256=file_sha256(crop)))
            deps.append(crop)
        height=760+math.ceil(len(objects)/3)*320;page=Image.new('RGB',(1440,height),'white');d=ImageDraw.Draw(page)
        d.text((10,10),f'{rid} planned={view["object_id"]}; all {len(truth)} labels; pending AI review',fill='black')
        annotated.thumbnail((1440,720));page.paste(annotated,(0,35))
        for j,o in enumerate(objects):
            crop=Image.open(o['crop']);crop.thumbnail((465,270));x=(j%3)*480;y=760+(j//3)*320
            d.text((x,y),f'{j} {o["scene_device_id"]} {o["class_name"]}',fill='black');page.paste(crop,(x,y+24))
        output=dest/f'{rid}.png';page.save(output);deps.extend([output,image])
        frames.append(dict(review_id=rid,view_id=v['view_id'],planned_object_id=view['object_id'],image_path=str(image.resolve()),image_sha256=v['image_sha256'],truth_sha256=object_sha256(v['truth']),objects=objects,page=str(output.resolve()),page_sha256=file_sha256(output)))
    return write_record(path,dict(status='twelve_full_frame_and_all_label_pages_pending_review',frames=frames,training_admitted=False,promotable=False,
        inputs={str(x.resolve()):file_sha256(x) for x in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');ap.add_argument('--evidence',action='store_true');ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args()
    if a.capture:print(asyncio.run(capture(a.attempt))['status'],flush=True)
    elif a.evidence:print(evidence()['status'],flush=True)
    else:print(freeze()['status'],'NO_CAPTURE')
