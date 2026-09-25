"""Unboxed structure evidence for the remaining legacy full-frame review.

Projection is spatial context only; no mask or automatic review decision.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior
from scripts.vision.diagnose_full_box_projection import translation,near_clip_box,project,SIGNS
from src.vision.canonical.plan import rotate


def render(source,image_path,page):
    tree=ET.parse(source['source_world'])
    camera=tree.find("./world/model[@name='canonical_camera']/link[@name='research_camera_link']")
    sensor=camera.find("sensor[@name='research_boxes']")
    q=source['actual_pose']['orientation'];rotation=np.column_stack([rotate(q,np.eye(3)[i]) for i in range(3)])
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8):raise ValueError('Nonorthogonal camera')
    optical=np.asarray(source['actual_pose']['position'])+rotation@(translation(camera)+translation(sensor))
    hfov=float(sensor.findtext('camera/horizontal_fov'));near=float(sensor.findtext('camera/clip/near'))
    width=int(sensor.findtext('camera/image/width'));height=int(sensor.findtext('camera/image/height'))
    im=Image.open(image_path).convert('RGB')
    if im.size!=(width,height):raise ValueError('Image/calibration dimensions differ')
    draw=ImageDraw.Draw(im);present={a['object_id'] for a in source['annotations']};candidates=[]
    for item in source['body_model_inventory']:
        if item['object_id'] in present:continue
        model=tree.find("./world/model[@name='%s']"%item['object_id']);links=model.findall('link')
        if len(links)!=1:raise ValueError('Unsupported multiple links')
        visual=links[0].find("visual[@name='body']")
        for node in (model,links[0],visual):
            pose=[float(x) for x in node.findtext('pose','0 0 0 0 0 0').split()]
            if any(abs(x)>1e-10 for x in pose[3:]):raise ValueError('Rotated geometry needs explicit transform')
        size=np.array([float(x) for x in visual.findtext('geometry/box/size','').split()])
        if size.shape!=(3,):continue
        center=translation(model)+translation(links[0])+translation(visual)
        points=near_clip_box((np.array([center+size*np.array(s)/2 for s in SIGNS])-optical)@rotation,near)
        if not len(points):continue
        pixels=(project(points,hfov,height/width)*np.array([1,-1])+1)*np.array([width,height])/2
        lo=np.maximum(pixels.min(0),[0,0]);hi=np.minimum(pixels.max(0),[width,height])
        if np.any(hi<=lo):continue
        box=[*lo.tolist(),*hi.tolist()];color='red' if item['labels'] else 'yellow'
        draw.rectangle(box,outline=color,width=3);draw.text(tuple(lo),item['object_id'],fill=color,stroke_width=1,stroke_fill='black')
        candidates.append(dict(object_id=item['object_id'],body_projection_xyxy=box,saved_target_labels=item['labels'],visibility='unknown'))
    for a in source['annotations']:
        draw.rectangle(a['bbox_xyxy'],outline='lime',width=2)
    im.thumbnail((1440,900));im.save(page)
    return candidates


def run():
    folder=OUT/'legacy-fullframe-review-v1';dest=folder/'evidence.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    names=['protocol','closeout-inventory-v2','legacy-review-links','legacy-world-source-trace','risk-reconciliation/review']
    paths=[OUT/(n+'.json') for n in names];p,q,legacy,trace,risk=[prior.read(path) for path in paths]
    for r in (p,q,legacy,trace,risk):prior.verify(r)
    pool={m['member_id']:m for m in p['members']};sources={m['member_id']:m for m in trace['members']}
    keep={m['member_id'] for m in q['members'] if m['stage']!='whole_frame_held'}
    selected={m['member_id']:m['decisions'][0]['event_id'].split('-')[0] for m in legacy['members'] if m['member_id'] in keep}
    for d in risk['decisions']:
        if d['review_id']=='R03':selected[d['member_id']]='R03'
    if len(selected)!=27:raise ValueError('Unexpected remaining legacy population')
    folder.mkdir(exist_ok=True);rows=[]
    for mid,rid in selected.items():
        source=sources[mid];m=pool[mid];page=folder/(rid+'.png')
        if prior.file_sha256(m['image_path'])!=m['image_sha256']:raise ValueError('Image changed')
        candidates=render(source,m['image_path'],page)
        rows.append(dict(member_id=mid,review_id=rid,page_path=str(page),page_sha256=prior.file_sha256(page),
            image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],source_world=source['source_world'],
            world_sha256=source['world_sha256'],unlabelled_body_projections=candidates,
            status='explicit_full_frame_review_required',pixel_visibility_certified=False))
        paths.extend([page,Path(m['image_path']),Path(m['label_path']),Path(source['source_world'])])
    paths.extend([Path(__file__).resolve(),Path(__file__).with_name('diagnose_full_box_projection.py')])
    return prior.frozen(dest,dict(status='27_full_frame_evidence_pages_not_approval',members=rows,
        legend='Green: existing label. Yellow: source body without target ID. Red: target body absent from labels. Projections ignore occlusion; even red is not proof of missing label.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
