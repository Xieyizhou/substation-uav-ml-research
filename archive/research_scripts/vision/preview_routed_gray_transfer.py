"""Freeze first active member of each material variant from actual RGB loader."""
from pathlib import Path
from types import SimpleNamespace
from PIL import Image,ImageDraw
from scripts.vision import routed_gray_transfer_control as run
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def main():
    p=run.checked(run.SOURCE/'protocol.json');key=run.SOURCE_KEYS[0];owner=SimpleNamespace(epoch=0)
    members={r['image_path']:r for r in p['pool_rows']};seen=set();rows=[]
    dest=run.OUT/'preview-v1';dest.mkdir(exist_ok=True)
    _,batches=run.runtime.baseline.loader(p,key,owner)
    for epoch in range(48):
        owner.epoch=epoch
        for j,b in enumerate(batches):
            pos=epoch*60+j*6;factors=p['contrast'][key][pos:pos+6]
            after,_=run.transform(b['img'],factors);before,_=run._contrast(b['img'],factors)
            for i,path in enumerate(b['im_file']):
                r=members[path]
                if factors[i]==1 or r['variant'] in seen:continue
                seen.add(r['variant']);canvas=Image.new('RGB',(1920,680),'white');d=ImageDraw.Draw(canvas)
                for col,t in enumerate((b['img'][i],before[i],after[i])):
                    canvas.paste(Image.fromarray(t.permute(1,2,0).numpy()),(640*col,40))
                d.text((5,5),r['member_id']+' | raw / routed contrast / plus gray',fill='black')
                page=dest/(r['variant']+'.png');canvas.save(page)
                rows.append(dict(member_id=r['member_id'],variant=r['variant'],position=pos+i,factor=factors[i],
                    image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],page=str(page.resolve()),page_sha256=file_sha256(page),
                    raw_tensor=run.runtime.tensor_hash(b['img'][i]),transformed_tensor=run.runtime.tensor_hash(after[i])))
            if len(seen)==6:break
        if len(seen)==6:break
    expected={'neutral','cool','warm','gray_all_body','gray_target_body','gray035'}
    if seen!=expected:raise ValueError('Material variant preview incomplete')
    paths=[run.SOURCE/'protocol.json',Path(__file__),Path(run.__file__)]+[Path(r['page']) for r in rows]
    return write_record(dest/'evidence.json',dict(rows=rows,status='actual_loader_examples_review_pending',training_admitted=False,promotable=False,
        inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(len(main()['rows']))
