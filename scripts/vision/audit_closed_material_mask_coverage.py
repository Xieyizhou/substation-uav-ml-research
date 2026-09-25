"""Compare all visible mapped equipment labels with full-frame box membership."""
from pathlib import Path
import re
import numpy as np
from PIL import Image
from scripts.vision.closed_body_material_pilot import OUT,read,verify,frozen,file_sha256


def coverage(mask,truth,mapping):
    labels={int(re.search(r'instance-(\d+)-',t['annotation_id'])[1]) for t in truth}
    visible=set(map(int,np.unique(mask[:,:,2])))-{0,255}
    if visible-{int(k) for k in mapping}:raise ValueError('Unmapped segmentation label')
    return sorted(visible-labels)


def main():
    dest=OUT/'mask-coverage.json'
    if dest.exists():verify(read(dest));return
    p=read(OUT/'protocol.json');q=read(OUT/'review-evidence.json');verify(p);verify(q)
    rows=[];extra=[];paths=[OUT/'protocol.json',OUT/'review-evidence.json',Path(__file__)];folder=OUT/'mask-evidence';folder.mkdir(exist_ok=True)
    for frame in p['frames']:
        rid=frame['review_ids'][0];truth=read(frame['source_receipt'])['truth']['objects'];mapping=frame['instance_mapping']
        for variant in ('control','warm','cool'):
            rp=OUT/'renders'/rid/variant/'replay'/rid/'attempt-01/receipt.json';receipt=read(rp);verify(receipt);paths.append(rp)
            stats=[]
            for n in receipt['selected_capture_indices']:
                mp=rp.parent/f'frame-{n}-mask.bin';rgb=rp.parent/f'frame-{n}-rgb.bin';paths.extend((mp,rgb))
                mask=np.frombuffer(mp.read_bytes(),dtype='u1').reshape(1080,1920,3)
                im=np.frombuffer(rgb.read_bytes(),dtype='u1').reshape(1080,1920,3)
                extras=coverage(mask,truth,mapping);stat={}
                for label in sorted(set(map(int,np.unique(mask[:,:,2])))-{0,255}):
                    ys,xs=np.where(mask[:,:,2]==label);box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                    stat[str(label)]=dict(pixel_count=len(xs),visible_bbox_xyxy=box,object_id=mapping[str(label)]['object_id'],category=mapping[str(label)]['category'])
                    if n==receipt['selected_capture_indices'][0] and (label in extras or rid=='T036' and label in (149,118,127)):
                        b=[max(0,box[0]-40),max(0,box[1]-40),min(1920,box[2]+40),min(1080,box[3]+40)]
                        raw=folder/f'{rid}-{variant}-{label}-rgb.png';Image.fromarray(im).crop(b).save(raw)
                        overlay=im.copy();sel=mask[:,:,2]==label;overlay[sel]=(overlay[sel].astype(float)*.5+np.array([255,0,255])*.5).astype('u1')
                        op=folder/f'{rid}-{variant}-{label}-overlay.png';Image.fromarray(overlay).crop(b).save(op);paths.extend((raw,op))
                stats.append(stat)
            if any(s!=stats[0] for s in stats):raise ValueError('Unstable per-equipment mask evidence')
            rows.append(dict(source_review_id=rid,variant=variant,receipt_identity=receipt['identity'],visible_equipment=stats[0],missing_box_labels=extras))
            for label in extras:extra.append(dict(source_review_id=rid,variant=variant,runtime_label=label,**stats[0][str(label)],status='label_coverage_gap_requires_investigation_not_automatic_relabel'))
    frozen(dest,dict(status='mask_box_coverage_audited',units=rows,anomalies=extra,training_ready=False,training_started=False,
        interpretation='Visible equipment membership versus annotated boxes, not component identity or automatic training quality. A visible edge fragment can require an explicit ignore/label policy, not automatic new annotation.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print(extra)


if __name__=='__main__':main()
