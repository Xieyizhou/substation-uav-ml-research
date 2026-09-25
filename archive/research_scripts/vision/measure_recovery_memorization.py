"""Training-image-only before/after instance recall at fixed confidence."""
import json
import sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1/memorization-v1'
    data=json.loads((base/'protocol.json').read_text());results={}
    paths={'baseline':ROOT/'models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt',
           'fit_default_nbs':base/'fit/weights/last.pt','fit_batchmatched':base/'fit-batchmatched/weights/last.pt'}
    for name,path in paths.items():
        model=YOLO(str(path));hits=total=predcount=0;frames=[]
        for item in data['selected']:
            row=item['row'];truth=row['truth']['objects']
            with Image.open(row['rgb_path']) as im:
                r=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
            used=set();hit=0
            for box,cls in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist()):
                eligible=[(iou(box,o['bbox_xyxy']),i) for i,o in enumerate(truth) if i not in used and o['class_name']==model.names[int(cls)]]
                overlap,index=max(eligible,default=(0,-1))
                if overlap>=.5:used.add(index);hit+=1
            hits+=hit;total+=len(truth);predcount+=len(r.boxes)
            frames.append({'view_id':row['view_id'],'hits':hit,'objects':len(truth),'predictions':len(r.boxes)})
        results[name]={'weights_sha256':file_sha256(path),'hits':hits,'objects':total,'predictions':predcount,'frames':frames}
    write_json(base/'training-image-comparison.json',{'results':results,'confidence':.37,'matching_iou':.5,
                'training_set_only':True,'promotable':False,'limits':['Same 18 training images for all measurements; not generalization.','Greedy same-class one-to-one instance matching.']})
    print(json.dumps({k:{a:b for a,b in v.items() if a!='frames'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
