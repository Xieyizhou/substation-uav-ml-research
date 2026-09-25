"""Bound actual pre-Gamma pixels to representative three-level review cards."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.closed_gamma_design import OUT,freeze,prior
from scripts.vision.preflight_closed_gamma import receipt
from scripts.vision.order_retention_runtime import make_dataset
from scripts.vision.brightness_transfer_runtime import brighten,pixel_hash
from scripts.vision.closed_gamma_runtime import transform

def main():
    p=freeze();root=OUT/'augmentation-review';dest=root/'evidence.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    r,rp=receipt('G900-7',p);seq=p['schedules']['G900-7'];rows={x['member_id']:x for x in p['pool_rows']}
    selected=[]
    for category in p['names']:
        for variant in ('original','cool'):
            choices=sorted({m for m in seq if rows[m].get('planned_category')==category and rows[m]['variant']==variant})
            if not choices:raise ValueError('Missing preview class/condition')
            selected.append(choices[0])
    selected+=sorted({m for m in seq if rows[m]['subset']=='hard_negative'})[:2]
    dataset=make_dataset(p,'G900-7');mapping={path:i for i,path in enumerate(dataset.im_files)};cards=[];deps=[OUT/'design.json',rp,Path(__file__).resolve()];root.mkdir(exist_ok=True)
    for n,member in enumerate(selected,1):
        m=rows[member];position=seq.index(member);raw=dataset.get_image_and_label(mapping[m['image_path']])['img']
        before=brighten(raw,p['brightness_factors']['G900-7'][position])
        if pixel_hash(before)!=r['gamma_log'][position]['before']:raise ValueError('Preview not actual pre-Gamma pixels')
        card=Image.new('RGB',(1200,275),'white');d=ImageDraw.Draw(card);outputs=[]
        for j,gamma in enumerate((1.,.8,1.2)):
            after=transform(before,gamma);im=Image.fromarray(after[:,:,::-1]);draw=ImageDraw.Draw(im);w,h=im.size
            for line in Path(m['label_path']).read_text().splitlines():
                c,x,y,bw,bh=map(float,line.split());draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='red',width=1)
            im.thumbnail((395,235));card.paste(im,(j*400,35));d.text((j*400+5,5),f'{n:02} {m.get("planned_category",m["subset"])} {m["variant"]} gamma={gamma}',fill='black')
            outputs.append(dict(gamma=gamma,pixel_sha256=pixel_hash(after)))
        cp=root/f'card-{n:02}.png';card.save(cp);deps.extend([cp,Path(m['image_path']),Path(m['label_path'])])
        cards.append(dict(id=f'G{n:02}',member_id=member,source_position=position,pre_gamma_sha256=pixel_hash(before),outputs=outputs,card_path=str(cp),card_sha256=prior.file_sha256(cp)))
    return prior.frozen(dest,dict(status='explicit_visual_observations_required',cards=cards,
        scope='Representative actual pre-Gamma inputs; all three factor outputs shown, only frozen scheduled factor used in each training exposure. Not full-pool semantic reapproval.',inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print('PREVIEW_CARDS',len(main()['cards']))
