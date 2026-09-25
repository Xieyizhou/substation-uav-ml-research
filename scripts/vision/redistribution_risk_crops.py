"""Original-pixel crops for explicitly selected unresolved full-label targets."""
from PIL import Image,ImageDraw
from pathlib import Path
from scripts.vision.prepare_redistribution_review import OUT,read,verify,frozen,file_sha256
TARGETS={'P05':4,'P09':3,'P17':1,'P31':2,'P33':3,'P40':1,'P44':4,'P46':0,'P47':2,'P53':2,'P54':1,'P56':0}
def main():
    e=read(OUT/'evidence.json');verify(e);paths=[OUT/'evidence.json',Path(__file__)];rows=[];tiles=[]
    for r in e['events']:
        eid=r['event_id']
        if eid not in TARGETS:continue
        i=TARGETS[eid];t=r['truth'][i];im=Image.open(r['member']['image_path']).convert('RGB');crop=im.crop(t['bbox_xyxy'])
        path=OUT/(eid+'-risk-crop.png');crop.save(path);paths.append(path)
        tile=Image.new('RGB',(400,360),'white');crop.thumbnail((390,315));tile.paste(crop,(5,40));ImageDraw.Draw(tile).text((8,8),eid+' '+t['class_name']+' line '+str(i),fill='black');tiles.append(tile)
        rows.append(dict(event_id=eid,label_line=i,truth=t,crop_path=str(path),crop_sha256=file_sha256(path)))
    for start in range(0,len(tiles),6):
        page=Image.new('RGB',(1200,720),'white')
        for j,t in enumerate(tiles[start:start+6]):page.paste(t,((j%3)*400,(j//3)*360))
        path=OUT/f'risk-page-{start//6+1}.png';page.save(path);paths.append(path)
    frozen(OUT/'risk-crops.json',dict(status='pending_explicit_review',targets=rows,inputs={str(x):file_sha256(x) for x in paths}))
if __name__=='__main__':main()
