"""Two named source projections; neither occlusion nor pixel certification."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.diagnose_full_box_projection import translation, near_clip_box, project, SIGNS
from src.vision.canonical.plan import rotate


def run():
    dest = OUT/'c01-background-projection.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    tp = OUT/'c01-source-correspondence.json'; trace = prior.read(tp); prior.verify(trace)
    f = trace['source_frame']; tree = ET.parse(f['source_world'])
    camera = tree.find("./world/model[@name='canonical_camera']/link[@name='research_camera_link']")
    sensor = camera.find("sensor[@name='research_boxes']")
    q = f['actual_pose']['orientation']; rotation = np.column_stack([rotate(q,np.eye(3)[i]) for i in range(3)])
    optical = np.asarray(f['actual_pose']['position']) + rotation @ (translation(camera)+translation(sensor))
    width = int(sensor.findtext('camera/image/width')); height = int(sensor.findtext('camera/image/height'))
    image = Image.open(f['source_image']).convert('RGB')
    if image.size != (width,height): raise ValueError('Sensor dimensions differ')
    draw = ImageDraw.Draw(image); rows = []
    op = Path(f['source_plan']).parent/'obstacles.json'; plan = prior.read(f['source_plan'])
    if prior.file_sha256(op) != plan['files']['obstacles.json']: raise ValueError('Taxonomy changed')
    for name in ['cabinet_center', 'control_building']:
        model = tree.find("./world/model[@name='%s']" % name); link = model.find('link'); visual = link.find("visual[@name='body']")
        size = np.array(list(map(float, visual.findtext('geometry/box/size').split())))
        center = translation(model)+translation(link)+translation(visual)
        vertices = np.array([center+size*np.array(s)/2 for s in SIGNS])
        points = near_clip_box((vertices-optical)@rotation, float(sensor.findtext('camera/clip/near')))
        pixels = (project(points,float(sensor.findtext('camera/horizontal_fov')),height/width)*[1,-1]+1)*[width/2,height/2]
        lo = np.maximum(pixels.min(0),[0,0]); hi = np.minimum(pixels.max(0),[width,height])
        if np.any(hi<=lo): raise ValueError('Named structure out of frame')
        box = [*lo.tolist(),*hi.tolist()]; draw.rectangle(box,outline='yellow',width=3); draw.text(tuple(lo),name,fill='yellow')
        assets = [x for x in prior.read(op)['obstacles'] if x['name']==name]
        if len(assets)!=1: raise ValueError('Ambiguous source asset')
        rows.append(dict(object_id=name, source_category=assets[0]['visual_category'], body_box_xyxy=box))
    image.thumbnail((1440,900)); page=OUT/'source-spatial-context/U51-background.png'; image.save(page)
    deps=[tp,op,page,Path(__file__).resolve(),Path(f['source_world']),Path(f['source_image'])]
    return prior.frozen(dest,dict(status='auxiliary_background_projections',rows=rows,page_path=str(page),pixel_visibility_certified=False,
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__': print(run()['page_path'])
