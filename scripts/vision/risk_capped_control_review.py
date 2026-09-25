"""Independent full-label evidence for risk-capped control; no training."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.check_risk_capped_redistribution import OUT as COUNTS,RUN,ROOT,ready,read,verify,frozen,file_sha256
from scripts.vision.structure_fit import OUT as FIT,truth_for
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence,validate_mapping
from scripts.vision.audit_redistribution_targets import resolve
from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world

OUT=RUN/'risk-capped-exposure-control-v1'

def build():
    p=ready();c=read(COUNTS/'counts.json');verify(c);verify(read(COUNTS/'completion.json'))
    trace=read(FIT/'member-source-trace.json');verify(trace)
    sources={r['member_id']:r for r in trace['rows']}
    increased=sorted({d['member_id'] for x in c['results'].values() for d in x['changes'] if d['delta']>0})
    idx={r['member_id']:r for r in p['pool_rows']};OUT.mkdir(exist_ok=True)
    paths=[COUNTS/'counts.json',COUNTS/'completion.json',FIT/'member-source-trace.json',Path(__file__),ROOT/'scripts/vision/audit_redistribution_targets.py']
    events=[];fulltiles=[];croptiles=[]
    for i,m in enumerate(increased,1):
        row=idx[m];source=sources[m];ip=Path(source['source_image']);rp=ip.parents[1]/(ip.parent.name+'.json');pp=ip.parents[2]/'plan/plan.json';wp=pp.parent/'world.sdf'
        for k in ('image','label'):
            if file_sha256(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('Stale member')
        equal_rgb(row['image_path'],ip);rec=read(rp);plan=read(pp);truth=truth_for(row)
        label_correspondence(truth,rec['truth']['objects']);mapping=instance_mapping(plan);validate_mapping(mapping)
        if file_sha256(wp)!=plan['files']['world.sdf']:raise ValueError('World hash conflict')
        mode=annotation_mode_from_world(wp)
        if mode!='full_2d':raise ValueError('Actual annotation mode mismatch')
        eid=f'C{i:02}';im=Image.open(ip).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full);labels=[]
        for j,t in enumerate(truth):
            label,obj=resolve(rec['raw_truth']['annotatedBox'],t,mapping);bid=f'{eid}-L{j}';b=t['bbox_xyxy']
            draw.rectangle(b,outline='lime',width=3);draw.text((max(0,b[0]),max(0,b[1])),bid+' '+t['class_name'],fill='red')
            x1,y1,x2,y2=b;crop=im.crop((max(0,int(x1)-12),max(0,int(y1)-12),min(im.width,int(x2)+13),min(im.height,int(y2)+13)))
            crop.thumbnail((480,290));tile=Image.new('RGB',(500,340),'white');tile.paste(crop,(0,45));ImageDraw.Draw(tile).text((5,5),bid+' '+t['class_name']+' '+obj['object_id'],fill='black')
            cp=OUT/(bid+'.png');tile.save(cp);croptiles.append(tile);paths.append(cp)
            labels.append(dict(event_id=bid,truth=t,runtime_label=label,object_id=obj['object_id'],crop_path=str(cp),crop_sha256=file_sha256(cp)))
        full.thumbnail((960,540));tile=Image.new('RGB',(960,580),'white');tile.paste(full,(0,35));ImageDraw.Draw(tile).text((5,5),eid+' '+row['subset'],fill='black')
        ep=OUT/(eid+'.png');tile.save(ep);fulltiles.append(tile)
        paths.extend([ip,rp,pp,wp,ep,Path(row['image_path']),Path(row['label_path'])])
        events.append(dict(event_id=eid,member=row,source_image=str(ip),source_receipt=str(rp),source_plan=str(pp),source_world=str(wp),actual_pose=rec['actual_pose'],
            mapping_basis='saved_plan_posthoc_not_historical_gate_certification',actual_mode=mode,labels=labels,evidence_path=str(ep),evidence_sha256=file_sha256(ep)))
    for name,tiles,w,h in [('full',fulltiles,960,580),('crops',croptiles,500,340)]:
        for start in range(0,len(tiles),6):
            page=Image.new('RGB',(w*2,h*3),'white')
            for j,t in enumerate(tiles[start:start+6]):page.paste(t,((j%2)*w,(j//2)*h))
            path=OUT/f'{name}-{start//6+1:02}.png';page.save(path);paths.append(path)
    frozen(OUT/'evidence.json',dict(status='full_label_review_pending',events=events,
        maximum_resolves=1,training_started=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('FROZEN',len(events),'IMAGES',sum(len(x['labels']) for x in events),'LABELS',str(OUT))

if __name__=='__main__':build()
