"""Generate actual-loader before/after evidence, never review decisions."""
from pathlib import Path
from types import SimpleNamespace
from PIL import Image,ImageDraw
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_scale_transform as scale
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=routed.OUT.parent/'routed-scale-transfer-control-v1'
EXPECTED=set(routed.MATERIAL)|{'original','negative'}

def main():
    import torch
    torch.set_num_threads(4)
    p=routed.checked(routed.OUT/'protocol.json');key=routed.KEYS[0];owner=SimpleNamespace(epoch=0)
    factors=scale.factors(p['pool_rows'],p['schedules'][key],7)
    members={r['image_path']:r for r in p['pool_rows']};seen=set();rows=[];dest=OUT/'preview-v1';dest.mkdir(parents=True,exist_ok=True)
    _,batches=routed.runtime.baseline.loader(p,key,owner)
    for epoch in range(48):
        owner.epoch=epoch
        for j,b in enumerate(batches):
            pos=epoch*60+j*6;before=dict(b);before['img'],_=routed.transform_runtime.transform(b['img'],p['contrast'][key][pos:pos+6]);after=scale.transform(before,factors[pos:pos+6])
            for i,path in enumerate(b['im_file']):
                r=members[path];group='negative' if not r['class_instances'] else r['variant']
                if group not in EXPECTED or group in seen or (group in routed.MATERIAL and factors[pos+i]!=.75):continue
                seen.add(group);canvas=Image.new('RGB',(1280,680),'white');draw=ImageDraw.Draw(canvas)
                draw.text((5,5),r['member_id']+' | routed contrast / scale; all boxes',fill='black')
                for col,t in enumerate((before,after)):
                    tile=Image.fromarray(t['img'][i].permute(1,2,0).numpy());d=ImageDraw.Draw(tile)
                    for box,cls in zip(t['bboxes'][t['batch_idx']==i],t['cls'][t['batch_idx']==i]):
                        x,y,w,h=box.tolist();d.rectangle(((x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640),outline='red',width=2)
                        d.text(((x-w/2)*640,(y-h/2)*640),str(int(cls.item())),fill='yellow')
                    canvas.paste(tile,(col*640,40))
                page=dest/(group+'.png');canvas.save(page)
                rows.append(dict(member_id=r['member_id'],group=group,position=pos+i,factor=factors[pos+i],image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],page=str(page.resolve()),page_sha256=file_sha256(page),before_tensor=routed.runtime.tensor_hash(before['img'][i]),after_tensor=routed.runtime.tensor_hash(after['img'][i]),before_boxes=before['bboxes'][before['batch_idx']==i].tolist(),after_boxes=after['bboxes'][after['batch_idx']==i].tolist()))
            if seen==EXPECTED:break
        if seen==EXPECTED:break
    if seen!=EXPECTED:raise ValueError('Missing preview groups')
    paths=[routed.OUT/'protocol.json',Path(__file__),Path(scale.__file__)]+[Path(r['page']) for r in rows]
    return write_record(dest/'evidence.json',dict(status='actual_scale_preview_review_pending',rows=rows,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(len(main()['rows']))
