"""Full-frame follow-up for uncertain training-box observations."""
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.build_switchgear_condition_review import OUT,read,save,file_sha256
IDS=[49,54,55,59,65,77,94,95,111,116,117,124,131,148,157,159,160,161,175,178,187,236,241,246,269,270,274,275,279,280,284,288,292]
def main():
    m=read(OUT/'manifest.json');selected=[m['items'][i-1] for i in IDS];pages=[]
    for start in range(0,len(selected),4):
        canvas=Image.new('RGB',(1200,760),'white');draw=ImageDraw.Draw(canvas)
        for n,r in enumerate(selected[start:start+4]):
            im=Image.open(r['image_path']).convert('RGB');s=min(590/im.width,330/im.height);im=im.resize((round(im.width*s),round(im.height*s)));d=ImageDraw.Draw(im);d.rectangle([v*s for v in r['bbox_xyxy']],outline='red',width=2);x=n%2*600;y=n//2*380;canvas.paste(im,(x,y+35));draw.text((x+5,y+5),r['review_id']+' full context; red = audited label',fill='black')
        p=OUT/f'context-{start//4+1:02}.jpg';canvas.save(p);pages.append(p)
    save(OUT/'full-context.json',dict(status='evidence_only',review_ids=[x['review_id'] for x in selected],inputs={str(p):file_sha256(p) for p in [OUT/'manifest.json',Path(__file__)]+pages}))
if __name__=='__main__':main()
