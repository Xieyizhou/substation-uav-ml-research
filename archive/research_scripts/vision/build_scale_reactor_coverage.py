"""Freeze all current reactor members and exposure; no inferred visual approval."""
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.freeze_order_fit_reviewed_dataset import run as dataset, OUT as SOURCE
from scripts.vision.freeze_reviewed_scale_control import OUT as TRAIN, KEYS, prior
from scripts.vision.train_reviewed_scale_control import complete
from scripts.vision.structure_fit import truth_for

OUT=TRAIN/'reactor-condition-coverage-v1'

def run():
    p=dataset();members=[m for m in p['members'] if m['class_instances'].get('reactor',0)]
    dest=OUT/'inventory.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    counts={};deps=[SOURCE/'reviewed-dataset-v1/manifest.json',Path(__file__).resolve()]
    for key in KEYS:
        c=complete(key);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        counts[key]=Counter(x['actual']);deps += [xp,TRAIN/'training'/key/'completion.json']
    rows=[];OUT.mkdir(exist_ok=True)
    for m in members:
        for kind in ('image','label'):
            path=Path(m[kind+'_path'])
            if prior.file_sha256(path)!=m[kind+'_sha256']:raise ValueError('Stale member')
            deps.append(path)
        truth=truth_for(m)
        if truth!=m['truth']:raise ValueError('Complete label drift')
        targets=[t for t in truth if t['class_name']=='reactor']
        if len(targets)!=m['class_instances']['reactor']:raise ValueError('Reactor count conflict')
        review=Path(m['full_frame_evidence']);prior.verify(prior.read(review));deps.append(review)
        for t in targets:
            rid=f'R{len(rows)+1:03}';im=Image.open(m['image_path']).convert('RGB');w,h=im.size
            crop=im.crop(t['bbox_xyxy']);draw=ImageDraw.Draw(im)
            for tt in truth:draw.rectangle(tt['bbox_xyxy'],outline='gray',width=2)
            draw.rectangle(t['bbox_xyxy'],outline='red',width=5)
            im.thumbnail((640,360));crop.thumbnail((310,350))
            card=Image.new('RGB',(960,420),'white');card.paste(im,(0,40));card.paste(crop,(645,40))
            ImageDraw.Draw(card).text((3,3),f'{rid} {m["member_id"]} {m["variant"]}',fill='black')
            path=OUT/(rid+'.png');card.save(path);deps.append(path)
            b=t['bbox_xyxy'];short=min(b[2]-b[0],b[3]-b[1])*640/max(w,h)
            rows.append(dict(review_id=rid,member_id=m['member_id'],truth=t,full_truth=truth,
                image_path=m['image_path'],image_sha256=m['image_sha256'],label_path=m['label_path'],label_sha256=m['label_sha256'],
                existing_full_frame_evidence=str(review),page=str(path),short_side_640=short,
                subset=m['subset'],variant=m['variant'],lineage_id=m['lineage_id'],
                resolved_historical_source=m.get('resolved_historical_source'),
                actual_image_exposures={k:counts[k][m['member_id']] for k in KEYS},
                visual_condition='pending_explicit_review',pixel_visibility_certified=False))
    return prior.frozen(dest,dict(status='complete_reactor_inventory_visual_condition_review_pending',
        members=len(members),reactor_targets=len(rows),rows=rows,
        registered_lineages=len({m['lineage_id'] for m in members}),independent_scene_count=None,
        scope='Existing quality approval is not proof of condition coverage. Registered lineage count is not independent-scene count.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':
    r=run();print(r['status'],r['members'],r['reactor_targets'],r['registered_lineages'])
