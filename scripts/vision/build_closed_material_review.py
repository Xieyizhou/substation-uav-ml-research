"""Full-label evidence and an empty explicit review queue, never approval."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scripts.vision.closed_body_material_pilot import OUT,read,verify,frozen,file_sha256


def main():
    dest=OUT/'review-evidence.json'
    if dest.exists():verify(read(dest));return
    p=read(OUT/'protocol.json');summary=read(OUT/'render-summary.json');verify(p);verify(summary)
    paths=[OUT/'protocol.json',OUT/'render-summary.json',Path(__file__)];events=[];font=ImageFont.load_default(size=18)
    folder=OUT/'review-evidence';folder.mkdir(exist_ok=True)
    for frame in p['frames']:
        rid=frame['review_ids'][0];truth=read(frame['source_receipt'])['truth']['objects']
        for variant in ('warm','cool'):
            unit=next(x for x in summary['units'] if x['review_id']==rid and x['variant']==variant)
            rp=Path(unit['receipt']);r=read(rp);verify(r)
            if r['status']!='material_render_passed_full_label_review_pending':raise ValueError('Material render gate failed')
            ip=rp.parent/'stable-1-rgb.png';im=Image.open(ip).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full)
            for i,t in enumerate(truth):
                b=t['bbox_xyxy'];draw.rectangle(b,outline='lime',width=3);draw.text((b[0]+3,b[1]+3),str(i),font=font,fill='yellow')
            fp=folder/f'{rid}-{variant}-full.png';full.save(fp);paths += [rp,ip,fp]
            for i,t in enumerate(truth):
                crop=folder/f'{rid}-{variant}-{i:02}.png';im.crop(tuple(t['bbox_xyxy'])).save(crop);paths.append(crop)
                events.append(dict(event_id=f'{rid}-{variant}-{i:02}',source_review_id=rid,variant=variant,member_id=frame['member_id'],lineage_id=frame['lineage_id'],
                    image_path=str(ip),image_sha256=file_sha256(ip),full_context_path=str(fp),full_context_sha256=file_sha256(fp),
                    crop_path=str(crop),crop_sha256=file_sha256(crop),source_truth=t,receipt_path=str(rp),receipt_identity=r['identity'],
                    decision=None,review_nature_required='AI-assisted',training_admitted=False,promotable=False))
    frozen(dest,dict(status='full_label_review_pending',images=8,labels=len(events),events=events,automatic_approvals_created=0,
        training_ready=False,training_started=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('REVIEW_QUEUE',len(events),'LABELS; NO_AUTOMATIC_APPROVALS')


if __name__=='__main__':main()
