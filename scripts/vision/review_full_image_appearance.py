"""Build per-frame evidence and exact instance-pair box checks; no decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.capture_full_image_appearance import OUT,ORIGINAL,read,save,file_sha256,verify_tree
from src.vision.canonical.plan import read_record
from src.ml.artifacts import object_sha256

def box_delta(original,current):
    if set(original)!=set(current):raise ValueError('Pair instance set differs')
    return max((abs(x-y) for k in original for x,y in zip(original[k],current[k])),default=0.)

def main():
    dest=OUT/'review/manifest.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    progress=OUT/'progress.json';verify_tree(progress);p=read(progress)
    if p['status']!='complete_pending_AI_review':raise ValueError('Incomplete capture')
    originals={f['view_id']:f for f in read(ORIGINAL/'review/decisions.json')['frames']}
    frames=[];inputs={};dest.parent.mkdir(exist_ok=False)
    for run in p['runs']:
        rp=Path(run['receipt_path']);r=read_record(rp);inputs[str(rp)]=file_sha256(rp)
        plan=read_record(Path(run['plan_path']));views={v['view_id']:v for v in plan['calibration_views']}
        mapping=r['collection_checks']['instance_mapping']
        for row in r['views']:
            if row['status']!='captured' or file_sha256(row['rgb_path'])!=row['image_sha256']:raise ValueError('Stale capture')
            v=views[row['view_id']];original=originals[v['original_view_id']];objects=[]
            image=Image.open(row['rgb_path']).convert('RGB');overlay=image.copy();d=ImageDraw.Draw(overlay)
            for b in row['raw_truth'].get('annotatedBox',[]):
                lo=b['box'].get('minCorner',{});hi=b['box'].get('maxCorner',{})
                box=[float(lo.get('x',0)),float(lo.get('y',0)),float(hi.get('x',0)),float(hi.get('y',0))]
                obj=mapping[str(b['label'])];planned=obj['object_id']==row['expected_object_id']
                d.rectangle(box,outline='red' if planned else 'lime',width=4)
                d.text((box[0],box[1]),obj['object_id'],fill='red' if planned else 'lime')
                objects.append(dict(**obj,runtime_label=b['label'],bbox_xyxy=box,planned=planned))
            delta=box_delta({o['object_id']:o['bbox_xyxy'] for o in original['objects']},{o['object_id']:o['bbox_xyxy'] for o in objects})
            fid=original['frame_id']+'-'+run['variant'];op=dest.parent/(fid+'.png');overlay.save(op)
            target=next(o for o in objects if o['planned']);x1,y1,x2,y2=target['bbox_xyxy'];cp=dest.parent/(fid+'-crop.png')
            image.crop((max(0,int(x1)-20),max(0,int(y1)-20),min(image.width,int(x2)+21),min(image.height,int(y2)+21))).save(cp)
            for path in (Path(row['rgb_path']),op,cp):inputs[str(path)]=file_sha256(path)
            frames.append(dict(frame_id=fid,original_frame_id=original['frame_id'],variant=run['variant'],view_id=row['view_id'],
              image_path=row['rgb_path'],image_sha256=row['image_sha256'],truth_sha256=object_sha256(row['raw_truth']),
              objects=objects,overlay_path=str(op),crop_path=str(cp),pair_bbox_max_delta=delta,
              pair_status='aligned' if delta<=1 else 'held_alignment',shared_source_group=v['shared_source_group'],review_status='pending'))
    for path in (progress,ORIGINAL/'review/decisions.json',Path(__file__)):inputs[str(path)]=file_sha256(path)
    save(dest,dict(status='pending_explicit_AI_review',frames=frames,frame_count=len(frames),box_count=sum(len(f['objects']) for f in frames),inputs=inputs))
    print('REVIEW_READY',len(frames),max(f['pair_bbox_max_delta'] for f in frames),flush=True)

if __name__=='__main__':main()
