"""Saved-geometry pixel-center reconstruction, not an instance-mask substitute."""
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.run_near_clip_build_test import OUT as BUILD,ROOT,read,verify,frozen,file_sha256,guard
from scripts.vision.verify_full_box_runtime_trace import OUT as RUNTIME

OUT=BUILD/'edge-raster-validation-v1'


def clip_polygon(poly):
    poly=[np.array(p,dtype=float) for p in poly]
    if any(p.shape!=(4,) or not np.isfinite(p).all() for p in poly):raise ValueError('Invalid homogeneous vertex')
    for plane in range(6):
        if not poly:break
        output=[];previous=poly[-1];before=previous[3]+(-1 if plane%2 else 1)*previous[plane//2]
        for current in poly:
            now=current[3]+(-1 if plane%2 else 1)*current[plane//2]
            if (before>=0)!=(now>=0):output.append(previous+before/(before-now)*(current-previous))
            if now>=0:output.append(current)
            previous=current;before=now
        poly=output
    return [p for p in poly if p[3]>1e-12]


def project_body(trace):
    if not np.allclose(trace['orientation_wxyz'],[1,0,0,0]):raise ValueError('Unsupported body rotation')
    signs=list(itertools.product((-1,1),repeat=3))
    world=np.array([np.array(trace['position'])+np.array(trace['scale'])*s/2 for s in np.array(signs)])
    view=np.array(trace['view_matrix']).reshape(4,4);projection=np.array(trace['projection_matrix']).reshape(4,4)
    vertices=np.c_[world,np.ones(8)]@view.T@projection.T
    polygons=[]
    for axis in range(3):
        for side in (-1,1):
            ids=[i for i,s in enumerate(signs) if s[axis]==side];ids=[ids[i] for i in (0,1,3,2)]
            clipped=clip_polygon(vertices[ids])
            if clipped:
                p=np.array(clipped);ndc=p[:,:2]/p[:,3,None]
                polygons.append((ndc*np.array([1,-1])+1)*np.array([960,540]))
    return polygons


def coverage(polygons,width=1920,height=1080):
    result=np.zeros((height,width),dtype=bool)
    if not polygons:return result
    vertices=np.concatenate(polygons)
    lo=np.maximum(np.floor(vertices.min(0)).astype(int),0)
    hi=np.minimum(np.ceil(vertices.max(0)).astype(int),[width,height])
    x0,y0=lo;x1,y1=hi
    yy,xx=np.mgrid[y0:y1,x0:x1];x=xx+.5;y=yy+.5
    for polygon in polygons:
        crosses=[(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]) for a,b in zip(polygon,np.roll(polygon,-1,axis=0))]
        c=np.array(crosses)
        result[y0:y1,x0:x1]|=np.all(c>=-1e-9,axis=0)|np.all(c<=1e-9,axis=0)
    return result


def bbox(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def edge_distance(point,polygons):
    distances=[]
    for polygon in polygons:
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            d=b-a
            if np.dot(d,d)==0:continue
            t=np.clip(np.dot(point-a,d)/np.dot(d,d),0,1)
            distances.append(float(np.linalg.norm(point-(a+t*d))))
    return min(distances)


def scanline_extent(polygons,y):
    crossings=[]
    for polygon in polygons:
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            if a[1]==b[1]:continue
            if min(a[1],b[1])<=y<=max(a[1],b[1]):crossings.append(float(a[0]+(y-a[1])/(b[1]-a[1])*(b[0]-a[0])))
    return [min(crossings),max(crossings)] if crossings else None


def main():
    guard()
    paths=[BUILD/'completion.json',RUNTIME/'completion.json']
    for p in paths:verify(read(p))
    folder=BUILD/'prefix-isolated/clipped/replay/T027/attempt-01';receipt=folder/'receipt.json';verify(read(receipt))
    trace_path=RUNTIME/'replay/T027/attempt-01/runtime-trace.jsonl'
    trace=next(json.loads(s) for s in trace_path.read_text().splitlines() if json.loads(s)['event']=='post_mesh_projection')
    polygons=project_body(trace);predicted=coverage(polygons)
    rows=[]
    for n in read(receipt)['selected_capture_indices']:
        mp=folder/f'frame-{n}-mask.bin';actual=np.frombuffer(mp.read_bytes(),dtype='u1').reshape(1080,1920,3)[:,:,2]==128
        ys,xs=np.where(predicted!=actual)
        rows.append(dict(frame=n,predicted_pixels=int(predicted.sum()),actual_pixels=int(actual.sum()),
            predicted_bbox=bbox(predicted),actual_bbox=bbox(actual),
            differing_pixels=[dict(x=int(x),y=int(y),predicted=bool(predicted[y,x]),actual=bool(actual[y,x]),distance_to_projected_edge_px=edge_distance(np.array([x+.5,y+.5]),polygons)) for y,x in zip(ys,xs)]))
        paths.append(mp)
    OUT.mkdir(exist_ok=True)
    rgb=folder/'frame-2-rgb.png';source=Image.open(rgb).convert('RGB')
    region=(45,1044,316,1080);zoom=source.crop(region).resize((1084,144),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(1084,210),'white');canvas.paste(zoom,(0,28));draw=ImageDraw.Draw(canvas)
    draw.text((6,6),'T027 bottom edge: red=full geometric x-min, cyan=first visible pixel column',fill='black')
    full=read(receipt)['records'][0]['full_boxes']['128']
    for x,color in [(full[0],'red'),(59,'cyan')]:draw.line([(int((x-45)*4),28),(int((x-45)*4),172)],fill=color,width=2)
    draw.text((6,183),'Geometry boundary y=1080; last pixel-center row y=1079.5. Analytical reconstruction only.',fill='black')
    evidence=OUT/'edge-evidence.png'
    if not evidence.exists():canvas.save(evidence)
    paths += [receipt,trace_path,rgb,evidence,Path(__file__)]
    dest=OUT/'raster-analysis.json'
    if dest.exists():verify(read(dest));print('VALID_RASTER_ANALYSIS_REUSED');return
    frozen(dest,dict(status='pixel_center_explains_bbox_gap_one_boundary_pixel_unresolved',
        geometric_bbox=[float(x) for x in np.r_[np.concatenate(polygons).min(0),np.concatenate(polygons).max(0)]],
        last_row_center_extent=scanline_extent(polygons,1079.5),polygons=[p.tolist() for p in polygons],comparisons=rows,
        evidence_kind='Analytical reconstruction from saved box geometry and observed runtime matrices, compared to real panoptic mask. Not independent rendered visibility certification.',
        unresolved='One pixel may reflect raster edge convention or numerical precision; no causal certification for that pixel without renderer-level coverage inspection.',
        training_ready=False,training_started=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(rows[0]);print('last-row',scanline_extent(polygons,1079.5))


if __name__=='__main__':main()
