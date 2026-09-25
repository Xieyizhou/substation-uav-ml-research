"""Evidence for the source-bound residual neutral-bridge errors, no decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.check_reviewed_hold_fit import OUT,prior,base

def main():
    p=prior.read(OUT/'protocol.json');s=prior.read(OUT/'summary.json');c=prior.read(OUT/'crosscheck.json')
    for x in (p,s,c):prior.verify(x)
    mids=sorted({d['member_id'] for d in s['details'] if d['common_exposed'] and not d['hit'] and d['member_id'] in c['bridge_sources'] and c['bridge_sources'][d['member_id']]['variant']=='neutral_bridge'})
    idx={r['member_id']:r for r in p['rows']};events=[];paths=[OUT/'protocol.json',OUT/'summary.json',OUT/'crosscheck.json',Path(__file__).resolve()]
    for n,mid in enumerate(mids,1):
        r=idx[mid];truth=base.truth_for(r);im=Image.open(r['image_path']).convert('RGB');full=im.copy();d=ImageDraw.Draw(full)
        for i,t in enumerate(truth):d.rectangle(t['bbox_xyxy'],outline='red',width=4);d.text(t['bbox_xyxy'][:2],str(i),fill='red')
        full.thumbnail((960,540));page=Image.new('RGB',(1460,650),'white');page.paste(full,(0,35))
        for i,t in enumerate(truth):
            crop=im.crop(t['bbox_xyxy']);crop.thumbnail((450,250));page.paste(crop,(980,35+i*290));ImageDraw.Draw(page).text((980,10+i*290),str(i)+' '+t['class_name'],fill='black')
        dest=OUT/'visual-crosscheck'/f'B{n:02}.png';page.save(dest);paths += [dest,Path(r['image_path'])]
        events.append(dict(event_id=f'B{n:02}',source=r,truth=truth,page=str(dest),page_sha256=prior.file_sha256(dest),results=[d for d in s['details'] if d['member_id']==mid]))
    prior.frozen(OUT/'bridge-residual-evidence.json',dict(events=events,status='awaiting_explicit_observation',inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(len(events))

if __name__=='__main__':main()
