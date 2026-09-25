"""Read-only experiment audit plus isolated evidence artifacts; no model selection."""
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.scale_endpoint_control import OUT,KEYS,prior,complete
from scripts.vision.evaluate_scale_endpoints import transitions
from scripts.vision import train_frozen_multiscale as old
from scripts.vision.diagnose_multiscale_errors import truth_key

DEST=OUT/'diagnosis-v1'

def main():
    DEST.mkdir(exist_ok=True)
    sp=OUT/'evaluation/summary.json';s=prior.read(sp);prior.verify(s)
    ip=OUT.parent/'material-control-feasibility-v1/initial-gate.json';identity=prior.read(ip);prior.verify(identity)
    lookup={}
    for x in identity['corrected_target_records']:
        k=(x['pair_id'],x['variant'],truth_key(x['truth']))
        if k in lookup:raise ValueError('Identity collision')
        lookup[k]=x
    paths=[sp,ip,Path(__file__).resolve()];rows={};audits={};fps=[]
    _,source,_=old.contract('fixed-7')
    np=Path(source['evaluation']['negative_review']);neg=prior.read(np);prior.verify(neg);paths.append(np)
    negative={(x['view_id'],x['variant']):x for x in neg['frames']}
    for key in KEYS:
        cell=complete(key);path=OUT/'evaluation'/f'{key}.json';r=prior.read(path);prior.verify(r);rows[key]=r
        seed=key.rsplit('-',1)[1];refp=old.OUT/'evaluation'/f'fixed-{seed}.json';ref=prior.read(refp);prior.verify(ref)
        tr=transitions(ref,r)
        if tr!=s['transitions'][key]:raise ValueError('Changed transitions')
        resolved=[]
        for x in tr:
            ident=lookup[(x['pair_id'],x['variant'],truth_key(x['truth']))]
            if x['truth']!=ident['truth']:raise ValueError('Changed complete truth')
            resolved.append({**x,'object_id':ident['object_id'],'source_identity':ident['evidence_id']})
        audits[key]=dict(resolved_transitions=resolved,
            counts=dict(Counter(f"{x['variant']}:{x['state']}" for x in tr)),
            loss_reasons=dict(Counter(f"{x['variant']}:{x['miss']['reason']}" for x in tr if x['state']=='loss')),
            matching_competition_losses=[x for x in resolved if x['state']=='loss' and x['miss']['formal_matching_competition']])
        paths += [path,refp,OUT/'training'/key/'completion.json']
        for x in r['negative_rows']:
            for pred in x['predictions']:
                frame=negative[(x['view_id'],x['variant'])]
                if frame['image_sha256']!=x['image_sha256'] or prior.file_sha256(frame['image_path'])!=x['image_sha256']:raise ValueError('Image mismatch')
                fps.append(dict(cell=key,frame=frame,prediction=pred))
    cards=[]
    for n,x in enumerate(fps,1):
        im=Image.open(x['frame']['image_path']).convert('RGB');box=x['prediction']['bbox_xyxy'];overlay=im.copy()
        ImageDraw.Draw(overlay).rectangle(box,outline='red',width=5);overlay.thumbnail((640,360))
        card=Image.new('RGB',(700,640),'white');d=ImageDraw.Draw(card)
        d.text((5,5),f"F{n:02} {x['cell']} {x['prediction']['class_name']} {x['prediction']['confidence']:.4f}",fill='black')
        card.paste(overlay,(5,25));crop=im.crop(tuple(map(round,box)));crop.thumbnail((680,230));card.paste(crop,(5,400))
        path=DEST/f'F{n:02}.png';card.save(path);paths += [path,Path(x['frame']['image_path'])]
        x.update(event_id=f'F{n:02}',page_path=str(path),page_sha256=prior.file_sha256(path));cards.append(card)
    pages=[]
    for start in range(0,len(cards),3):
        sheet=Image.new('RGB',(2100,640),'white')
        for j,card in enumerate(cards[start:start+3]):sheet.paste(card,(j*700,0))
        path=DEST/f'page-{start//3+1:02}.png';sheet.save(path);paths.append(path);pages.append(str(path))
    prior.frozen(DEST/'audit.json',dict(status='numerical_and_identity_audit_complete_visual_review_pending',audits=audits,
        false_positives=fps,pages=pages,selected_candidate=None,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('AUDITED',len(audits),'CELLS;',len(fps),'FP boxes;',len(pages),'pages')

if __name__=='__main__':main()
