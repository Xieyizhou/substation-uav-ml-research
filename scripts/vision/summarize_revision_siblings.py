"""Technical summary and local instance overlays; does not issue quality approvals."""
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.vision.replay_revision_siblings import OUT,base


def inspect(record, folder):
    if record['status']!='existing_pose_technical_checks_passed' or not record['process_cleanup_complete']:raise ValueError('Incomplete replay')
    if len(record['records'])!=3:raise ValueError('Three stable frames required')
    counts=[]
    for r in record['records']:
        if not r['rgb_exact'] or r['skew_ms']>33.334001 or r['lost_labels'] or r['added_labels']!=['128'] or max(r['historical_deltas'].values())>1:raise ValueError('Alignment or label gate failed')
        mask=np.frombuffer((folder/f'frame-{r["capture_index"]}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        ys,xs=np.where(mask[:,:,2]==128)
        if not len(xs):raise ValueError('Target instance absent')
        counts.append(dict(pixels=len(xs),bbox_half_open=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]))
    if any(x!=counts[0] for x in counts):raise ValueError('Unstable target mask')
    return counts[0]


def main():
    dest=OUT/'evidence.json'
    if dest.exists():base.verify(base.read(dest));return
    pp=OUT/'run-receipt.json';base.verify(base.read(pp));p=base.read(pp)
    if p['status']!='runs_complete' or len(p['results'])!=2:raise ValueError('Two complete units required')
    paths=[pp,Path(__file__)];rows=[]
    for u in p['results']:
        rp=Path(u['receipt']);r=base.read(rp);base.verify(r);folder=rp.parent;stats=inspect(r,folder)
        n=r['records'][0]['capture_index'];rgb=Image.open(folder/f'frame-{n}-rgb.png').convert('RGB')
        mask=np.frombuffer((folder/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        overlay=np.array(rgb);overlay[mask[:,:,2]==128]=[255,0,255]
        canvas=Image.new('RGB',(1200,500),'white')
        for i,im in enumerate((rgb,Image.fromarray(overlay))):
            crop=im.crop((0,1000,400,1080)).resize((1200,240));canvas.paste(crop,(0,i*250))
        page=OUT/(u['review_id']+'-instance.png');canvas.save(page)
        rows.append(dict(**u,**stats,page=str(page),original_pixel_instance_certified=True,
            training_quality_approved=False,max_original_box_delta=max(max(x['historical_deltas'].values()) for x in r['records'])))
        paths += [rp,page]
    base.frozen(dest,dict(status='two_independent_original_pixel_instance_evidence_units',members=rows,
        training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':main()
