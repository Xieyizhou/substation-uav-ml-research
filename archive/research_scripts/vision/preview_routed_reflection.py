"""Deterministic actual-loader reflection examples, with all boxes drawn."""
from pathlib import Path
from types import SimpleNamespace
from PIL import Image,ImageDraw
from scripts.vision import routed_reflection_control as run
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

EXPECTED={'neutral','cool','warm','gray_all_body','gray_target_body','gray035','original','negative'}

def main():
    p=run.checked(run.SOURCE/'protocol.json');key=run.SOURCE_KEYS[0];owner=SimpleNamespace(epoch=0)
    members={r['image_path']:r for r in p['pool_rows']};seen=set();rows=[];dest=run.OUT/'preview-v1';dest.mkdir(parents=True,exist_ok=True)
    _,batches=run.runtime.baseline.loader(p,key,owner);flags=run.positions(7)
    for epoch in range(48):
        owner.epoch=epoch
        for j,b in enumerate(batches):
            pos=epoch*60+j*6;before=dict(b);before['img'],_=run.augmentation.transform(b['img'],p['contrast'][key][pos:pos+6]);after=run.reflect(before,flags[pos:pos+6])
            for i,path in enumerate(b['im_file']):
                r=members[path];group='negative' if not r['class_instances'] else r['variant']
                if not flags[pos+i] or group not in EXPECTED or group in seen:continue
                seen.add(group);canvas=Image.new('RGB',(1280,680),'white');draw=ImageDraw.Draw(canvas)
                draw.text((5,5),r['member_id']+' | routed contrast / reflected + complete boxes',fill='black')
                for col,t in enumerate((before,after)):
                    tile=Image.fromarray(t['img'][i].permute(1,2,0).numpy());d=ImageDraw.Draw(tile)
                    for box,cls in zip(t['bboxes'][t['batch_idx']==i],t['cls'][t['batch_idx']==i]):
                        x,y,w,h=box.tolist();d.rectangle(((x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640),outline='red',width=2)
                        d.text(((x-w/2)*640,(y-h/2)*640),str(int(cls.item())),fill='yellow')
                    canvas.paste(tile,(col*640,40))
                page=dest/(group+'.png');canvas.save(page)
                rows.append(dict(member_id=r['member_id'],group=group,position=pos+i,image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],
                    page=str(page.resolve()),page_sha256=file_sha256(page),before_tensor=run.runtime.tensor_hash(before['img'][i]),after_tensor=run.runtime.tensor_hash(after['img'][i])))
            if seen==EXPECTED:break
        if seen==EXPECTED:break
    if seen!=EXPECTED:raise ValueError('Missing preview groups '+str(EXPECTED-seen))
    paths=[run.SOURCE/'protocol.json',Path(__file__),Path(run.__file__)]+[Path(r['page']) for r in rows]
    return write_record(dest/'evidence.json',dict(status='actual_loader_reflection_preview_pending',rows=rows,training_admitted=False,promotable=False,
        inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(len(main()['rows']))
