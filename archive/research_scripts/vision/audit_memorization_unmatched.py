"""Reproduce training-image matching and retain unmatched predictions for review."""
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1/memorization-v1'
    source=base/'protocol.json';comparison=base/'training-image-comparison.json'
    expected=json.loads(comparison.read_text())['results']['fit_batchmatched']
    weights=base/'fit-batchmatched/weights/last.pt'
    assert file_sha256(weights)==expected['weights_sha256']
    inputs={str(p):file_sha256(p) for p in (source,comparison,weights,Path(__file__))}
    model=YOLO(str(weights));queue=[];missed=[];all_frames=[];hits=predictions=0
    for item in json.loads(source.read_text())['selected']:
        row=item['row'];truth=row['truth']['objects'];path=Path(row['rgb_path'])
        inputs[str(path)]=file_sha256(path);assert inputs[str(path)]==row['image_sha256']
        with Image.open(path) as im:
            result=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
        preds=[{'bbox_xyxy':b,'class_name':model.names[int(k)],'confidence':c} for b,k,c in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist())]
        used=set();unmatched=[]
        for rank,p in enumerate(preds):
            overlap,index=max([(iou(p['bbox_xyxy'],o['bbox_xyxy']),i) for i,o in enumerate(truth) if i not in used and o['class_name']==p['class_name']],default=(0,-1))
            if overlap>=.5:used.add(index)
            else:unmatched.append((rank,p))
        for rank,p in unmatched:
            same=max([iou(p['bbox_xyxy'],o['bbox_xyxy']) for o in truth if o['class_name']==p['class_name']],default=0)
            best,index=max([(iou(p['bbox_xyxy'],o['bbox_xyxy']),i) for i,o in enumerate(truth)],default=(0,-1))
            category='duplicate_same_class' if same>=.5 else 'wrong_class_on_target' if best>=.5 else 'localization_overlap' if best>=.1 else 'background_candidate'
            queue.append({'review_index':len(queue)+1,'view_id':row['view_id'],'image_path':str(path),'prediction_rank':rank,
                          'prediction':p,'truth':truth,'best_iou':best,'same_class_best_iou':same,
                          'best_truth_index':index,'geometric_category':category})
        for i,o in enumerate(truth):
            if i not in used:missed.append({'view_id':row['view_id'],'truth':o,'image_path':str(path)})
        hits+=len(used);predictions+=len(preds)
        all_frames.append({'view_id':row['view_id'],'predictions':preds,'matched_truth_indices':sorted(used)})
    assert hits==expected['hits'] and predictions==expected['predictions'] and len(queue)==19
    output=base/'unmatched-audit-v1';output.mkdir(exist_ok=True)
    visual=ROOT/'outputs/research/ml_training_recovery_v1/unmatched-audit-v1';visual.mkdir(parents=True,exist_ok=True);sheets={}
    for start in range(0,len(queue),6):
        page=Image.new('RGB',(1280,1200),'#202020');draw=ImageDraw.Draw(page)
        for j,row in enumerate(queue[start:start+6]):
            x,y=(j%2)*640,(j//2)*400
            with Image.open(row['image_path']) as im:page.paste(im.resize((640,360)),(x,y+40))
            p=row['prediction'];draw.text((x+3,y+3),f'{row["review_index"]}: {p["class_name"]} {p["confidence"]:.3f} {row["geometric_category"]}',fill='white')
            draw.text((x+3,y+20),f'IoU {row["best_iou"]:.3f}; orange prediction, green all truth',fill='white')
            for obj,color in [(o,'lime') for o in row['truth']]+[(p,'orange')]:
                a,b,c,d=obj['bbox_xyxy'];draw.rectangle((x+a/3,y+40+b/3,x+c/3,y+40+d/3),outline=color,width=2)
                draw.text((x+a/3,y+40+b/3),obj['class_name'],fill=color)
        p=visual/f'page-{start//6+1:02}.jpg';page.save(p,quality=95);sheets[str(p)]=file_sha256(p)
    report={'inputs':inputs,'queue':queue,'missed_targets':missed,'frames':all_frames,'visuals':sheets,
            'matched':hits,'predictions':predictions,'geometric_counts':dict(Counter(r['geometric_category'] for r in queue)),
            'status':'reproduced_pending_visual_review','training_set_only':True,'promotable':False,
            'limits':['Geometric categories are candidate explanations until visually reviewed.','Same training images, no generalization claim.']}
    report['identity']=object_sha256(report);write_json(output/'report.json',report)
    print(json.dumps({'matched':hits,'predictions':predictions,'unmatched':len(queue),'missed':len(missed),'categories':report['geometric_counts']},indent=2))


if __name__=='__main__':main()
