"""Saved body-box spatial context only; never visibility or label certification."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.diagnose_full_box_projection import translation, near_clip_box, project, SIGNS
from src.vision.canonical.plan import rotate


def run():
    folder=OUT/'source-spatial-context'; dest=folder/'evidence.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    tp=OUT/'legacy-world-source-trace.json'; ep=OUT/'remaining-review/evidence.json'
    trace=prior.read(tp);e=prior.read(ep)
    prior.verify(trace);prior.verify(e)
    idx={r['member_id']:r for r in trace['members']}
    deps=[tp,ep,Path(__file__).resolve()];rows=[];folder.mkdir(exist_ok=True)
    for event in e['events']:
        source=idx.get(event['member_id'])
        if not source:continue
        row=dict(review_id=event['review_id'],member_id=event['member_id'],pixel_visibility_certified=False)
        try:
            wp=Path(source['source_world']);ip=Path(source['source_image']);deps.extend([wp,ip])
            tree=ET.parse(wp)
            camera=tree.find("./world/model[@name='canonical_camera']/link[@name='research_camera_link']")
            sensor=camera.find("sensor[@name='research_boxes']")
            q=source['actual_pose']['orientation']
            rotation=np.column_stack([rotate(q,np.eye(3)[i]) for i in range(3)])
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8):raise ValueError('invalid_rotation')
            optical=np.asarray(source['actual_pose']['position'])+rotation@(translation(camera)+translation(sensor))
            hfov=float(sensor.findtext('camera/horizontal_fov'));near=float(sensor.findtext('camera/clip/near'))
            width=int(sensor.findtext('camera/image/width'));height=int(sensor.findtext('camera/image/height'))
            im=Image.open(ip).convert('RGB')
            if im.size!=(width,height):raise ValueError('sensor_image_size_mismatch')
            draw=ImageDraw.Draw(im);projected=[]
            for item in source['body_model_inventory']:
                model=tree.find("./world/model[@name='%s']"%item['object_id'])
                links=model.findall('link')
                if len(links)!=1:raise ValueError('unsupported_links')
                visual=links[0].find("visual[@name='body']")
                size=np.array([float(x) for x in visual.findtext('geometry/box/size','').split()])
                if size.shape!=(3,):continue
                center=translation(model)+translation(links[0])+translation(visual)
                vertices=np.array([center+size*np.array(s)/2 for s in SIGNS])
                points=near_clip_box((vertices-optical)@rotation,near)
                if not len(points):continue
                ndc=project(points,hfov,height/width)
                pixels=(ndc*np.array([1,-1])+1)*np.array([width,height])/2
                lo=np.maximum(pixels.min(0),[0,0]);hi=np.minimum(pixels.max(0),[width,height])
                if np.any(hi<=lo):continue
                box=[*lo.tolist(),*hi.tolist()]
                color='yellow' if not item['labels'] else 'lime'
                draw.rectangle(box,outline=color,width=3)
                draw.text(tuple(lo),item['object_id'],fill=color,stroke_width=1,stroke_fill='black')
                projected.append(dict(object_id=item['object_id'],saved_labels=item['labels'],body_box_xyxy=box))
            im.thumbnail((1440,900));page=folder/(event['review_id']+'.png');im.save(page);deps.append(page)
            row.update(status='auxiliary_body_projection_only',projected=projected,page_path=str(page))
        except (ValueError,AttributeError,KeyError) as exc:
            row.update(status='unsupported_source_projection',reason=str(exc))
        rows.append(row)
    return prior.frozen(dest,dict(status='spatial_context_not_instance_masks',rows=rows,
        limits='Analytical body AABBs do not account for occlusion, certify visible pixels, or establish background semantics. Yellow means no saved target label, not an approved negative.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    r=run();print(len(r['rows']),sum(x['status']=='auxiliary_body_projection_only' for x in r['rows']))
