"""Render certified same-frame target masks; no review decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.replay_development_visibility_gaps import OUT,prior

def main():
    pp,rp=OUT/'protocol.json',OUT/'replay-receipt.json';p,r=prior.read(pp),prior.read(rp)
    for x in (p,r):prior.verify(x)
    if r['status']!='replays_complete_review_pending':raise ValueError('Replay not complete')
    paths=[pp,rp,Path(__file__).resolve()];events=[]
    folder=OUT/'evidence';folder.mkdir(exist_ok=True)
    for f,unit in zip(p['frames'],r['results'],strict=True):
        receipt=Path(unit['receipt']);c=prior.read(receipt);prior.verify(c);paths.append(receipt)
        if c['status']!='capture_technical_checks_passed' or not c['process_cleanup_complete']:raise ValueError('Invalid replay')
        if len(c['records'])!=3 or not all(x['rgb_exact_vs_original'] and x['skew_ms']<=33.334001 and max(x['historical_deltas'].values(),default=0)<=1 for x in c['records']):raise ValueError('Alignment failed')
        i=c['records'][0]['capture_index']
        for e in f['events']:
            targets=[x for x in c['targets'] if x['event_id']==e['event_id']]
            if len(targets)!=3 or len({(x['visible_pixels'],tuple(x['visible_bbox'] or [])) for x in targets})!=1:raise ValueError('Target mask unstable')
            ip=Path(f['source_image']);op=receipt.parent/f'{i}-{e["event_id"]}-overlay.png';im=Image.open(ip).convert('RGB');over=Image.open(op).convert('RGB')
            box=e['truth']['bbox_xyxy'];crop=im.crop(box);maskcrop=over.crop(box)
            page=Image.new('RGB',(1440,850),'white');full=over.copy();full.thumbnail((960,540));page.paste(full,(0,25))
            for n,img in enumerate((crop,maskcrop)):
                img.thumbnail((680,260));page.paste(img,(n*720,585))
            ImageDraw.Draw(page).text((0,5),f"{e['event_id']} {e['object_id']} pixels={targets[0]['visible_pixels']} exact RGB; stable 3",fill='black')
            dest=folder/(e['event_id']+'.png');page.save(dest);paths += [dest,ip,op]
            events.append(dict(**e,source_condition=f['source_visual_condition'],image_path=str(ip),image_sha256=prior.file_sha256(ip),
                page=str(dest),page_sha256=prior.file_sha256(dest),visible_pixels=targets[0]['visible_pixels'],visible_bbox=targets[0]['visible_bbox'],
                original_pixel_evidence_certified=True,replay_receipt=str(receipt),skew_ms=[x['skew_ms'] for x in c['records']],
                bbox_max_delta=max(max(x['historical_deltas'].values(),default=0) for x in c['records'])))
    prior.frozen(OUT/'evidence.json',dict(status='certified_masks_visual_review_pending',events=events,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print([(x['event_id'],x['visible_pixels'],x['visible_bbox']) for x in events])

if __name__=='__main__':main()
