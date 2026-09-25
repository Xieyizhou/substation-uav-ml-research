"""Whole-frame scale/padding probe, no clipping or training admission."""
import argparse
from pathlib import Path
from PIL import Image
from scripts.vision.diagnose_reviewed_target_fit import OUT as ANCHOR,freeze as anchors,prior,runtime,DEST,KEYS

OUT=DEST/'scale-padding-probe-v1'
SCALES=(1.,.75,.5)
def transform(im,truth,scale):
    if scale not in SCALES:raise ValueError('Unfrozen scale')
    w,h=im.size;nw,nh=round(w*scale),round(h*scale);ox,oy=(w-nw)//2,(h-nh)//2
    if scale==1:out=im.copy()
    else:
        out=Image.new('RGB',(w,h),(114,114,114));out.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(ox,oy))
    rows=[]
    for t in truth:
        x1,y1,x2,y2=t['bbox_xyxy']
        # Decimal YOLO normalization may reconstruct a boundary at -9.6e-8 px.
        # Preserve coordinates, allowing only sub-micropixel arithmetic error.
        if not (-1e-6<=x1<x2<=w+1e-6 and -1e-6<=y1<y2<=h+1e-6):raise ValueError('Invalid source box')
        rows.append(dict(t,bbox_xyxy=[x1*nw/w+ox,y1*nh/h+oy,x2*nw/w+ox,y2*nh/h+oy]))
    return out,rows

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    a=anchors();OUT.mkdir(exist_ok=True);members=[];deps=[ANCHOR/'completion.json',ANCHOR/'protocol.json',Path(__file__).resolve(),Path(runtime.__file__).resolve()]
    for m in a['members']:
        im=Image.open(m['image_path']).convert('RGB')
        for scale in SCALES:
            out,truth=transform(im,m['truth'],scale);mid=m['member_id']+f'@scale{scale:g}'
            if scale==1:path=Path(m['image_path'])
            else:path=OUT/f'image-{len(members):02}.png';out.save(path)
            members.append(dict(m,member_id=mid,parent_member_id=m['member_id'],scale=scale,image_path=str(path),image_sha256=prior.file_sha256(path),truth=truth,
                derived_diagnostic_only=True,training_admitted=False,promotable=False))
            deps += [path,Path(m['image_path']),Path(m['label_path'])]
    return prior.frozen(dest,dict(status='scale_padding_probe_frozen',members=members,models=a['models'],environment=a['environment'],inference=a['inference'],anchor_targets=a['targets'],
        transform=dict(scales=list(SCALES),interpolation='PIL_BILINEAR',fill_rgb=[114,114,114],placement='center_integer_offset',clipping=False),
        interpretation='Scale changes accompany interpolation, padding and context occupancy; not pure camera-distance causality. Derivatives were not training members.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))

def finish(p):
    results=[];deps=[OUT/'protocol.json'];targets={t['member_id']:t for t in p['anchor_targets']}
    for k in KEYS:
        path=OUT/(k+'.json');r=prior.read(path);runtime.validate(r,k,p);deps.append(path)
        for m,row in zip(p['members'],r['rows']):
            t=targets[m['parent_member_id']];j=next(i for i,t0 in enumerate(m['truth']) if t0['label_line_index']==t['truth']['label_line_index'])
            results.append(dict(model=k,target_id=t['target_id'],scale=m['scale'],member_id=m['member_id'],hit=any(x['truth_index']==j for x in row['matches']),
                miss=next((x for x in row['misses'] if x['truth_index']==j),None),full_truth=len(row['truth']),full_matches=len(row['matches'])))
    aggregates={str(scale):dict(target_hits=sum(x['hit'] for x in results if x['scale']==scale),target_events=sum(x['scale']==scale for x in results),
        full_truth=sum(x['full_truth'] for x in results if x['scale']==scale),full_matches=sum(x['full_matches'] for x in results if x['scale']==scale)) for scale in SCALES}
    dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='scale_padding_probe_complete',aggregates=aggregates,rows=results,inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze();print('FROZEN',len(p['members']),flush=True)
    if a.infer:
        runtime.OUT=OUT
        for k in KEYS:runtime.infer(k,p)
        print(finish(p)['aggregates'],flush=True)
