"""Read-only provenance and independent box projection audit for three edge labels."""
import sys,math,itertools
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.record_switchgear_condition_review import OUT as REVIEW,read,save,file_sha256,verify_tree
from src.vision.canonical.plan import rotate
from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world
OUT=REVIEW/'edge-source-audit-v1'
BASE=ROOT/'data/research/ml_training_recovery_v1'
SOURCES=[('T049','stratified-expansion-v1/medium','32095105d2029d07d357e674d53efe52fd3c1a7fe8a72ca425373833c5d912d7'),('T054','stratified-expansion-v1/complex','8028a788c683457db4f5da215f2520e4753b2e9d9a9a3ca85221c51ff352d8fb'),('T148','visual-augmentation-240-v1/positive-capture-v1/runs/regular-positive','92a5c2e4d4eea974cd3d055be1561e41deb6e1ca63491157418c15c9dbce3504')]

def hull(points):
    pts=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    halves=[]
    for sequence in (pts,pts[::-1]):
        half=[]
        for p in sequence:
            while len(half)>=2 and cross(half[-2],half[-1],p)<=0:half.pop()
            half.append(p)
        halves.append(half[:-1])
    return halves[0]+halves[1]

def clipped_area(poly,width,height):
    for axis,bound,sign in ((0,0,1),(0,width,-1),(1,0,1),(1,height,-1)):
        result=[]
        if not poly:return 0.
        for a,b in zip(poly,poly[1:]+poly[:1]):
            ia=sign*(a[axis]-bound)>=0;ib=sign*(b[axis]-bound)>=0
            if ia:result.append(a)
            if ia!=ib:
                t=(bound-a[axis])/(b[axis]-a[axis]);result.append(tuple(a[j]+t*(b[j]-a[j]) for j in range(2)))
        poly=result
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])))/2

def pose(node):
    values=list(map(float,(node.findtext('pose') or '0 0 0 0 0 0').split()))
    if any(abs(v)>1e-10 for v in values[3:]):raise ValueError('Rotated static visual unsupported')
    return values[:3]

def main():
    verify_tree(REVIEW/'completion.json');manifest=read(REVIEW/'manifest.json');lookup={r['review_id']:r for r in manifest['items']};inputs={str(REVIEW/'completion.json'):file_sha256(REVIEW/'completion.json'),str(Path(__file__)):file_sha256(Path(__file__))};results=[]
    for rid,folder,view in SOURCES:
        base=BASE/folder;pp=base/'plan/plan.json';wp=base/'plan/world.sdf';tp=base/'capture'/f'{view}.json';cp=base/'capture/collection-receipt.json'
        plan=read(pp);captured=read(tp);receipt=read(cp);row=lookup[rid];tree=ET.parse(wp)
        for q in (pp,wp,tp,cp,Path(row['image_path']),Path(row['label_path']),Path(captured['rgb_path'])):inputs[str(q)]=file_sha256(q)
        if file_sha256(wp)!=plan['files']['world.sdf']:raise ValueError('World hash changed')
        if file_sha256(captured['rgb_path'])!=captured['image_sha256']:raise ValueError('Source image changed')
        source=Image.open(captured['rgb_path']).convert('RGB');training=Image.open(row['image_path']).convert('RGB')
        if source.size!=training.size or source.tobytes()!=training.tobytes():raise ValueError('Training pixels differ from source')
        w,h=source.size;cls,x,y,bw,bh=map(float,Path(row['label_path']).read_text().splitlines()[row['label_line_index']].split());box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
        candidates=[]
        for raw in captured['raw_truth']['annotatedBox']:
            a=raw['box']['minCorner'];b=raw['box']['maxCorner'];bb=[a.get('x',0),a.get('y',0),b.get('x',0),b.get('y',0)];candidates.append((max(abs(u-v) for u,v in zip(box,bb)),raw['label'],bb))
        delta,label,raw_box=min(candidates);mapping=instance_mapping(plan)
        if delta>1e-4 or mapping[label]['category']!='switchgear' or cls!=1:raise ValueError('Export/instance mismatch')
        name=mapping[label]['object_id'];model=next(m for m in tree.iter('model') if m.get('name')==name);mp=pose(model)
        camera=next(m for m in tree.iter('model') if m.get('name')=='canonical_camera');link=camera.find("link[@name='research_camera_link']");rgb=link.find("sensor[@name='research_rgb']/camera")
        q=captured['actual_pose']['orientation'];offset=rotate(q,pose(link));center=[a+b for a,b in zip(captured['actual_pose']['position'],offset)];inv=[-q[0],-q[1],-q[2],q[3]];fx=w/(2*math.tan(float(rgb.findtext('horizontal_fov'))/2));visuals=[];all_points=[]
        for l in model.findall('link'):
            lp=pose(l)
            for v in l.findall('visual'):
                vp=pose(v);size=v.findtext('geometry/box/size')
                if size is None:raise ValueError('Non-box visual unsupported')
                dims=list(map(float,size.split()));points=[]
                for signs in itertools.product((-1,1),repeat=3):
                    world=[mp[i]+lp[i]+vp[i]+signs[i]*dims[i]/2 for i in range(3)];a,b,c=rotate(inv,[world[i]-center[i] for i in range(3)])
                    if a<=0:raise ValueError('Near-plane geometry needs clipping')
                    points.append((w/2-fx*b/a,h/2-fx*c/a))
                poly=hull(points);all_points+=points;visuals.append(dict(name=v.get('name'),projected_hull=poly,image_intersection_area_px2=clipped_area(poly,w,h)))
        projected=[min(p[0] for p in all_points),min(p[1] for p in all_points),max(p[0] for p in all_points),max(p[1] for p in all_points)]
        results.append(dict(review_id=rid,source_capture=str(tp),source_world=str(wp),source_image=captured['rgb_path'],training_image=row['image_path'],label_line_index=row['label_line_index'],pixels_identical=True,label_export_max_error_px=delta,runtime_label=label,scene_object_id=name,planned_object_id=captured['expected_object_id'],is_planned_target=name==captured['expected_object_id'],raw_box=raw_box,annotation_declared=plan['annotation_mode'],annotation_world=annotation_mode_from_world(wp),receipt_actual_mode=receipt.get('actual_annotation_mode'),check_version=captured.get('check_version'),world_hash=inputs[str(wp)],optical_center=center,visuals=visuals,projected_visual_envelope=projected,
            projection_limitations='Independent ideal pinhole projection of saved static box visuals at saved pose; no renderer replay, distortion, occlusion or pixel-instance mask. Projection is corroboration, not visibility certification.',visibility='unknown_pending',decision='hold_no_relabel'))
    save(OUT/'audit.json',dict(status='source_export_verified_visibility_pending',results=results,training_admitted=False,promotable=False,inputs=inputs));print([(r['review_id'],r['scene_object_id'],r['label_export_max_error_px'],r['raw_box'],r['projected_visual_envelope'],[(v['name'],v['image_intersection_area_px2']) for v in r['visuals']]) for r in results])
if __name__=='__main__':main()
