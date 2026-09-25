"""Deterministically freeze twelve source views and own-box review evidence."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.establish_material_view_candidates import OUT,HISTORY,prior
from src.vision.canonical.plan import pose_close

def main():
    ip=OUT/'source-inventory.json';inventory=prior.read(ip);prior.verify(inventory)
    cp=HISTORY/'coverage-census.json';c=prior.read(cp);prior.verify(c)
    seen=[r['actual_pose'] for r in c['members'] if r.get('actual_pose') and any(r['actual_exposures'].values())]
    rows=[r for r in inventory['records'] if r['status']=='source_trace_passed_review_required' and r['lighting_id']=='light_normal' and not any(pose_close(r['actual_pose'],p) for p in seen)]
    grouped={}
    for r in rows:grouped.setdefault(r['lineage_id'],[]).append(r)
    representatives=[]
    for group,items in grouped.items():
        r=min(items,key=lambda x:({'material_original':0,'mat_cool_gray':1,'mat_desaturated_green':2,'mat_warm_oxide':3}.get(x['material_id'],9),x['source_member_id']))
        p=prior.read(r['source_plan']);v=next(x for k in ('calibration_views','pilot_views') for x in p.get(k,[]) if x['view_id']==r['source_view_id'])
        representatives.append(dict(r,distance=v['distance'],bearing=v['bearing']))
    selected=[]
    for category in ('capacitor_bank','switchgear','reactor','transformer'):
        rs=sorted([r for r in representatives if r['category']==category],key=lambda x:(x['distance'],x['object_id'],x['bearing'],x['source_view_id']))
        if len(rs)<3:raise ValueError('Fewer than three traceable poses: '+category)
        # First deterministic member in each distance-rank third; no image/model quality ranking.
        selected += [dict(rs[i*len(rs)//3],distance_rank_band=i) for i in range(3)]
    paths=[ip,cp,Path(__file__).resolve()];events=[];pages=[];(OUT/'source-review').mkdir(exist_ok=True)
    for n,r in enumerate(selected,1):
        rid=f'N{n:02}';r['source_review_id']=rid
        truths=r['source_record']['truth']['objects'];im=Image.open(r['source_image']).convert('RGB')
        canvas=Image.new('RGB',(1400,460+210*((len(truths)+1)//2)),'white');d=ImageDraw.Draw(canvas)
        over=im.copy();od=ImageDraw.Draw(over)
        for j,t in enumerate(truths):od.rectangle(t['bbox_xyxy'],outline='red',width=3);od.text((t['bbox_xyxy'][0],t['bbox_xyxy'][1]),str(j),fill='red')
        over.thumbnail((760,420));canvas.paste(over,(0,30));d.text((0,5),f"{rid} {r['category']} {r['material_id']} distance={r['distance']}",fill='black')
        import re
        for j,t in enumerate(truths):
            label=str(int(re.search(r'instance-(\d+)-',t['annotation_id'])[1]));identity=r['instance_mapping'][label]
            if identity['category']!=t['class_name']:raise ValueError('Truth category/instance mismatch')
            crop=im.crop(tuple(t['bbox_xyxy']));dest=OUT/'source-review'/f'{rid}-{j:02}.png';crop.save(dest);paths.append(dest)
            crop.thumbnail((680,175));x=(j%2)*700;y=460+(j//2)*210;canvas.paste(crop,(x,y+25))
            d.text((x,y),f"{rid}-{j:02} {identity['object_id']}",fill='black')
            events.append(dict(event_id=f'{rid}-{j:02}',source_review_id=rid,truth=t,object_id=identity['object_id'],runtime_label=label,
                crop_path=str(dest),crop_sha256=prior.file_sha256(dest),image_sha256=r['image_sha256']))
        page=OUT/'source-review'/f'{rid}.png';canvas.save(page);paths.append(page)
        r.update(page_path=str(page),page_sha256=prior.file_sha256(page),review_status='awaiting_explicit_full_label_review',source_independent=False)
        pages.append(str(page))
    prior.frozen(OUT/'source-review.json',dict(status='twelve_sources_frozen_review_required',sources=selected,events=events,pages=pages,
        selection_rule='Unexposed registered group and no matching resolved exposed pose; normal light, original appearance preferred within group; first in each distance rank third',
        limitations=['Same complex training/development layout/assets','Source material may already be altered; not automatically original V arm','Resolved-pose comparison is not a complete audit of unresolved historical sources'],
        training_ready=False,training_started=False,collection_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('SOURCES',len(selected),'TRUTHS',len(events))
    for r in selected:print(r['source_review_id'],r['category'],r['distance'],r['material_id'],r['lineage_id'])

if __name__=='__main__':main()
