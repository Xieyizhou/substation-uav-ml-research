"""Current complete-label evidence for historical adverse-review reconciliation."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior


def run():
    p=freeze();rp=OUT/'risk-history.json';r=prior.read(rp);prior.verify(r)
    idx={m['member_id']:m for m in p['members']};folder=OUT/'risk-reconciliation';folder.mkdir(exist_ok=True)
    events=[];deps=[rp,OUT/'protocol.json',Path(__file__).resolve()]
    for n,risk in enumerate((v for v in r['members'] if v['currently_exposed']),1):
        m=idx[risk['member_id']];image=Image.open(m['image_path']).convert('RGB');truth=m['truth']
        page=Image.new('RGB',(1500,510+300*((len(truth)+2)//3)),'white');draw=ImageDraw.Draw(page)
        full=image.copy();d=ImageDraw.Draw(full)
        for j,t in enumerate(truth):
            d.rectangle(t['bbox_xyxy'],outline='red',width=3);d.text(t['bbox_xyxy'][:2],str(j),fill='red')
        full.thumbnail((900,470));page.paste(full,(0,30));draw.text((5,5),f'R{n:02} '+m['member_id'],fill='black')
        for j,t in enumerate(truth):
            x=(j%3)*500;y=510+(j//3)*300
            crop=image.crop(tuple(t['bbox_xyxy']));crop.thumbnail((485,260));page.paste(crop,(x,y+30))
            draw.text((x,y),f'{j} '+t['class_name'],fill='black')
        dest=folder/f'R{n:02}.png';page.save(dest);deps.extend([dest,Path(m['image_path']),Path(m['label_path'])])
        events.append(dict(review_id=f'R{n:02}',member_id=m['member_id'],image_sha256=m['image_sha256'],
            label_sha256=m['label_sha256'],truth=truth,page_path=str(dest),page_sha256=prior.file_sha256(dest),
            historical_adverse_records=risk['adverse_records'],status='explicit_current_review_required'))
    dest=folder/'evidence.json'
    if dest.exists():v=prior.read(dest);prior.verify(v);return v
    return prior.frozen(dest,dict(status='current_full_label_risk_pages_pending',events=events,
        pixel_visibility_certified=False,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':print(len(run()['events']))
