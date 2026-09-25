"""Contact pages for inspecting all frozen labels in both variants; no decisions."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scripts.vision.closed_body_material_pilot import OUT,read,verify,frozen,file_sha256


def main():
    path=OUT/'review-pages.json'
    if path.exists():verify(read(path));return
    q=read(OUT/'review-evidence.json');verify(q);dest=OUT/'review-pages';dest.mkdir(exist_ok=True)
    font=ImageFont.load_default(size=18);pages=[];paths=[OUT/'review-evidence.json',Path(__file__)]
    for rid in ('T020','T036','T023','T027'):
        rows={v:[x for x in q['events'] if x['source_review_id']==rid and x['variant']==v] for v in ('warm','cool')}
        full=Image.new('RGB',(1600,500),'white');d=ImageDraw.Draw(full)
        for j,v in enumerate(('warm','cool')):
            im=Image.open(rows[v][0]['full_context_path']);im.thumbnail((800,450));full.paste(im,(j*800,40));d.text((j*800+4,4),rid+' '+v,font=font,fill='black')
        fp=dest/f'{rid}-full.png';full.save(fp);pages.append(str(fp));paths.append(fp)
        for start in range(0,len(rows['warm']),4):
            page=Image.new('RGB',(1000,880),'white');d=ImageDraw.Draw(page)
            for i in range(start,min(start+4,len(rows['warm']))):
                for j,v in enumerate(('warm','cool')):
                    row=rows[v][i];im=Image.open(row['crop_path']);im.thumbnail((490,180));x=j*500;y=(i-start)*220
                    page.paste(im,(x,y+35));d.text((x+3,y+3),row['event_id']+' '+row['source_truth']['class_name'],font=font,fill='black')
            fp=dest/f'{rid}-labels-{start:02}.png';page.save(fp);pages.append(str(fp));paths.append(fp)
    frozen(path,dict(status='awaiting_explicit_inspection',pages=pages,inputs={str(p):file_sha256(p) for p in paths}))
    print('\n'.join(pages))


if __name__=='__main__':main()
