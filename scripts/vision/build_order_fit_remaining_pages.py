"""Freeze complete-label audit of members not covered by existing strict links."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior


def run():
    folder=OUT/'remaining-review';dest=folder/'evidence.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    p=freeze();active=set().union(*(set(v) for v in p['actual_exposures'].values()))
    ip=OUT/'quality-isolation.json';iso=prior.read(ip);prior.verify(iso)
    held={m['member_id'] for m in iso['members'] if m['status']!='requires_complete_quality_and_role_validation'}
    linked=set();deps=[OUT/'protocol.json',ip,Path(__file__).resolve()]
    for name in ('cohort-review-links','negative-review-links','early-positive-review-links','legacy-review-links'):
        path=OUT/(name+'.json');r=prior.read(path);prior.verify(r);deps.append(path)
        linked|={m['member_id'] for m in r['members']}
    selected=[m for m in p['members'] if m['member_id'] in active-held-linked]
    folder.mkdir(exist_ok=True);events=[]
    for n,m in enumerate(selected,1):
        im=Image.open(m['image_path']).convert('RGB');truth=m['truth'];full=im.copy();d=ImageDraw.Draw(full)
        for i,t in enumerate(truth):d.rectangle(t['bbox_xyxy'],outline='red',width=3);d.text(t['bbox_xyxy'][:2],str(i),fill='red')
        full.thumbnail((1000,560));page=Image.new('RGB',(1500,600+300*((len(truth)+2)//3)),'white');page.paste(full,(0,30));d=ImageDraw.Draw(page)
        d.text((0,0),f'U{n:02} '+m['member_id'],fill='black')
        for i,t in enumerate(truth):
            x=(i%3)*500;y=600+(i//3)*300;crop=im.crop(tuple(t['bbox_xyxy']));crop.thumbnail((485,265));page.paste(crop,(x,y+30));d.text((x,y),f'{i} '+t['class_name'],fill='black')
        card=folder/f'U{n:02}.png';page.save(card);deps.extend([card,Path(m['image_path']),Path(m['label_path'])])
        events.append(dict(review_id=f'U{n:02}',member_id=m['member_id'],image_sha256=m['image_sha256'],
            label_sha256=m['label_sha256'],truth=truth,page_path=str(card),page_sha256=prior.file_sha256(card)))
    return prior.frozen(dest,dict(status='remaining_current_full_label_review_pending',events=events,
        selection_rule='All currently exposed, non-isolated members without strict existing full-label links; no model-score selection.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':print(len(run()['events']))
