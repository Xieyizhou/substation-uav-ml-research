"""Evidence for newly increased members after the only permitted resolve."""
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.risk_capped_control_review import OUT as CONTROL,FIT,ROOT,ready,read,verify,frozen,file_sha256,truth_for,equal_rgb,label_correspondence,validate_mapping,resolve,instance_mapping,annotation_mode_from_world

OUT=CONTROL/'post-resolve-review-v1'

def main():
    p=ready();r=read(CONTROL/'resolve-01.json');e=read(CONTROL/'evidence.json');trace=read(FIT/'member-source-trace.json')
    for d in (r,e,trace):verify(d)
    if r['additional_resolves_used']!=1:raise ValueError('Unexpected solve history')
    known={x['member']['member_id'] for x in e['events']};inc=set()
    for seed,x in r['results'].items():
        old=Counter(p['schedules'][f'reference-450-{seed}']);inc|={m for m,n in x['counts'].items() if n>old[m]}
    new=sorted(inc-known);idx={x['member_id']:x for x in p['pool_rows']};sources={x['member_id']:x for x in trace['rows']}
    OUT.mkdir(exist_ok=True);frames=[];tiles=[];paths=[CONTROL/'resolve-01.json',CONTROL/'evidence.json',FIT/'member-source-trace.json',Path(__file__)]
    for i,m in enumerate(new,1):
        row=idx[m];ip=Path(sources[m]['source_image']);rp=ip.parents[1]/(ip.parent.name+'.json');pp=ip.parents[2]/'plan/plan.json';wp=pp.parent/'world.sdf'
        for k in ('image','label'):
            if file_sha256(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('Stale member')
        equal_rgb(ip,row['image_path']);rec=read(rp);plan=read(pp);truth=truth_for(row);label_correspondence(truth,rec['truth']['objects'])
        mapping=instance_mapping(plan);validate_mapping(mapping)
        if file_sha256(wp)!=plan['files']['world.sdf'] or annotation_mode_from_world(wp)!='full_2d':raise ValueError('World mismatch')
        eid=f'N{i:02}';im=Image.open(ip).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full);labels=[]
        for j,t in enumerate(truth):
            bid=f'{eid}-L{j}';label,obj=resolve(rec['raw_truth']['annotatedBox'],t,mapping);b=t['bbox_xyxy'];draw.rectangle(b,outline='lime',width=3);draw.text((max(0,b[0]),max(0,b[1])),bid+' '+t['class_name'],fill='red')
            x1,y1,x2,y2=b;crop=im.crop((max(0,int(x1)-12),max(0,int(y1)-12),min(im.width,int(x2)+13),min(im.height,int(y2)+13)));crop.thumbnail((480,290))
            tile=Image.new('RGB',(500,340),'white');tile.paste(crop,(0,45));ImageDraw.Draw(tile).text((5,5),bid+' '+t['class_name'],fill='black');cp=OUT/(bid+'.png');tile.save(cp);tiles.append(tile);paths.append(cp)
            labels.append(dict(event_id=bid,truth=t,object_id=obj['object_id'],runtime_label=label,crop_path=str(cp),crop_sha256=file_sha256(cp)))
        ep=OUT/(eid+'.png');full.save(ep);paths.extend([ip,rp,pp,wp,ep,Path(row['image_path']),Path(row['label_path'])])
        frames.append(dict(event_id=eid,member=row,labels=labels,evidence_path=str(ep),evidence_sha256=file_sha256(ep),source_image=str(ip),source_receipt=str(rp),source_plan=str(pp),source_world=str(wp),actual_pose=rec['actual_pose'],mapping_basis='saved_plan_posthoc_not_original_gate_certificate'))
    for start in range(0,len(tiles),6):
        page=Image.new('RGB',(1000,1020),'white')
        for j,t in enumerate(tiles[start:start+6]):page.paste(t,((j%2)*500,(j//2)*340))
        path=OUT/f'crops-{start//6+1:02}.png';page.save(path);paths.append(path)
    frozen(OUT/'evidence.json',dict(status='post_resolve_review_pending',events=frames,training_started=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('NEW_IMAGES',len(frames),'LABELS',len(tiles))

if __name__=='__main__':main()
