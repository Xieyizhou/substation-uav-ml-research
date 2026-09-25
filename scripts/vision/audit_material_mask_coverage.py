"""Independent fail-closed full-target mask coverage audit; never edits labels."""
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.vision.expand_material_view_n05 import OUT,prior


def coverage(mask,box_labels,mapping):
    if mask.ndim!=3 or mask.shape[2]!=3:raise ValueError('Not RGB panoptic mask')
    labels,counts=np.unique(mask[:,:,2],return_counts=True)
    observed={int(k):int(v) for k,v in zip(labels,counts) if k not in (0,255)}
    if not observed:raise ValueError('No mapped instance labels; cannot certify encoding')
    if set(observed)-set(map(int,mapping)):raise ValueError('Unresolved instance label')
    if len(box_labels)!=len(set(box_labels)):raise ValueError('Duplicate box instance')
    for label in observed:
        if np.any(np.all(mask[mask[:,:,2]==label,:2]==0,axis=1)):raise ValueError('Category-like mask lacks instance count')
    missing=[]
    for label in sorted(set(observed)-set(box_labels)):
        ys,xs=np.where(mask[:,:,2]==label)
        missing.append(dict(runtime_label=label,object_id=mapping[str(label)]['object_id'],
            visible_pixels=observed[label],visible_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]))
    return dict(status='blocked_unboxed_target_pixels' if missing else 'visible_targets_have_boxes',
        missing_targets=missing,visible_pixels_by_label=observed)


def main():
    p=prior.read(OUT/'protocol.json');prior.verify(p);f=p['frame'];dest=OUT/'coverage-audit';dest.mkdir(exist_ok=True)
    labels=[x['runtime_label'] for x in f['events']];rows=[];paths=[OUT/'protocol.json',Path(__file__)]
    for variant in ('source-control','original','warm','cool'):
        rp=OUT/variant/'replay/N05/attempt-01/receipt.json';r=prior.read(rp);prior.verify(r);paths.append(rp)
        for n in range(1,4):
            root=rp.parent/'first-stable-window';mp=root/f'frame-{n}-mask.bin'
            mask=np.fromfile(mp,dtype='u1').reshape(1080,1920,3);result=coverage(mask,labels,f['instance_mapping'])
            rows.append(dict(variant=variant,frame=n,**result));paths.append(mp)
            if n==1:
                ip=root/'frame-1-rgb.png';rgb=np.asarray(Image.open(ip).convert('RGB')).copy();paths.append(ip)
                for missing in result['missing_targets']:
                    hit=mask[:,:,2]==missing['runtime_label'];rgb[hit]=(rgb[hit]*.5+np.array([255,0,255])*.5).astype('u1')
                    op=dest/f'{variant}-{missing["runtime_label"]}.png';Image.fromarray(rgb).save(op);paths.append(op)
                    crop=dest/f'{variant}-{missing["runtime_label"]}-crop.png'
                    Image.fromarray(rgb).crop(missing['visible_bbox_xyxy']).save(crop);paths.append(crop)
    prior.frozen(dest/'audit.json',dict(status='N05_whole_group_held_for_unboxed_transformer',records=rows,
        training_ready=False,labels_modified=False,automatic_label_repair=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print([(r['variant'],r['frame'],r['missing_targets']) for r in rows])

if __name__=='__main__':main()
