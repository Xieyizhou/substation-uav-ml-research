"""Auxiliary projection of ordinary cabinet components at U33's saved pose."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.diagnose_full_box_projection import translation, near_clip_box, project, SIGNS
from src.vision.canonical.plan import rotate


def run():
    dest=OUT/'u33-component-projection.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    tp=OUT/'legacy-world-source-trace.json';ep=OUT/'remaining-review/evidence.json'
    t,e=prior.read(tp),prior.read(ep);prior.verify(t);prior.verify(e)
    event=next(x for x in e['events'] if x['review_id']=='U33')
    s=next(x for x in t['members'] if x['member_id']==event['member_id'])
    tree=ET.parse(s['source_world']);camera=tree.find("./world/model[@name='canonical_camera']/link[@name='research_camera_link']")
    sensor=camera.find("sensor[@name='research_boxes']");q=s['actual_pose']['orientation']
    rotation=np.column_stack([rotate(q,np.eye(3)[i]) for i in range(3)])
    optical=np.asarray(s['actual_pose']['position'])+rotation@(translation(camera)+translation(sensor))
    image=Image.open(s['source_image']).convert('RGB');w,h=image.size;draw=ImageDraw.Draw(image);rows=[]
    model=tree.find("./world/model[@name='cabinet_2']");link=model.find('link')
    for visual in link.findall('visual'):
        size=np.array(list(map(float,visual.findtext('geometry/box/size').split())))
        center=translation(model)+translation(link)+translation(visual)
        vertices=np.array([center+size*np.array(sign)/2 for sign in SIGNS])
        points=near_clip_box((vertices-optical)@rotation,float(sensor.findtext('camera/clip/near')))
        if not len(points):continue
        pixels=(project(points,float(sensor.findtext('camera/horizontal_fov')),h/w)*[1,-1]+1)*[w/2,h/2]
        lo=np.maximum(pixels.min(0),[0,0]);hi=np.minimum(pixels.max(0),[w,h])
        if np.any(hi<=lo):continue
        box=[*lo.tolist(),*hi.tolist()];draw.rectangle(box,outline='yellow',width=2)
        draw.text(tuple(lo),'cabinet_2/'+visual.get('name'),fill='yellow')
        rows.append(dict(object_id='cabinet_2',component=visual.get('name'),box_xyxy=box))
    page=OUT/'source-spatial-context/U33-components.png';image.save(page)
    deps=[tp,ep,page,Path(__file__).resolve(),Path(s['source_world']),Path(s['source_image'])]
    return prior.frozen(dest,dict(status='auxiliary_component_projection_not_mask',member_id=s['member_id'],
        rows=rows,page_path=str(page),pixel_visibility_certified=False,
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['rows'])
