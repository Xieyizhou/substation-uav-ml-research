"""Expose every full-image box and target crop; never synthesize review decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.prepare_appearance_recovery_candidates import OUT,read,save,file_sha256,verify_tree,ROOT
from src.vision.canonical.plan import read_record
from src.ml.artifacts import object_sha256

def main():
    target=OUT/'review/manifest.json'
    if target.exists():verify_tree(target);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'pilot-protocol.json');progress=read(OUT/'capture-progress.json')
    if progress['status']!='complete_pending_AI_review':raise ValueError('Incomplete pilot capture')
    folder=target.parent;folder.mkdir(parents=True,exist_ok=False);frames=[];inputs={}
    for run in progress['runs']:
        path=Path(run['receipt_path']);receipt=read_record(path);inputs[str(path)]=file_sha256(path)
        mapping=receipt['collection_checks']['instance_mapping']
        for row in receipt['views']:
            if row['status']!='captured' or file_sha256(row['rgb_path'])!=row['image_sha256']:raise ValueError('Stale/incomplete capture')
            inputs[row['rgb_path']]=row['image_sha256'];image=Image.open(row['rgb_path']).convert('RGB');overlay=image.copy();d=ImageDraw.Draw(overlay)
            boxes=row['raw_truth'].get('annotatedBox',row['raw_truth'].get('annotated_box',[]));objects=[];targetbox=None
            for index,b in enumerate(boxes):
                identity=mapping[str(b['label'])];lo=b['box'].get('minCorner',{});hi=b['box'].get('maxCorner',{});box=[float(lo.get('x',0)),float(lo.get('y',0)),float(hi.get('x',0)),float(hi.get('y',0))]
                planned=identity['object_id']==row['expected_object_id']
                d.rectangle(box,outline='red' if planned else 'lime',width=4);d.text((max(0,box[0]),max(0,box[1])),f"{index}:{identity['object_id']}",fill='red' if planned else 'lime')
                objects.append(dict(index=index,**identity,runtime_label=b['label'],bbox_xyxy=box,planned=planned,review_status='pending'))
                if planned:
                    if targetbox is not None:raise ValueError('Duplicate planned target box')
                    targetbox=box
            if targetbox is None:raise ValueError('Missing planned target')
            stem=run['variant']+'-'+row['expected_category'];op=folder/(stem+'-overlay.png');cp=folder/(stem+'-crop.png')
            overlay.save(op);x1,y1,x2,y2=targetbox;image.crop((max(0,int(x1)-20),max(0,int(y1)-20),min(image.width,int(x2)+21),min(image.height,int(y2)+21))).save(cp)
            for p in (op,cp):inputs[str(p)]=file_sha256(p)
            frames.append(dict(variant=run['variant'],view_id=row['view_id'],category=row['expected_category'],object_id=row['expected_object_id'],image_path=row['rgb_path'],image_sha256=row['image_sha256'],truth_sha256=object_sha256(row['raw_truth']),overlay_path=str(op),crop_path=str(cp),objects=objects,review_status='pending'))
    for p in (OUT/'pilot-protocol.json',OUT/'capture-progress.json',Path(__file__)):inputs[str(p)]=file_sha256(p)
    save(target,dict(status='pending_explicit_AI_review',frames=frames,frame_count=len(frames),box_count=sum(len(r['objects']) for r in frames),inputs=inputs))
    print('REVIEW_READY',len(frames),sum(len(r['objects']) for r in frames),folder,flush=True)

if __name__=='__main__':main()
