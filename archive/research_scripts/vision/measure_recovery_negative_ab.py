"""Compare positive fit and negative-augmented fit on the fixed diagnostic set."""
import json
import sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1';mem=base/'memorization-v1';ab=base/'negative-ab-v1'
    protocol=json.loads((mem/'protocol.json').read_text());ab_protocol=json.loads((ab/'protocol.json').read_text())
    paths={'positive_only':mem/'fit-batchmatched/weights/last.pt','negative_augmented':ab/'fit/weights/last.pt'}
    results={}
    for name,path in paths.items():
        model=YOLO(str(path));positive_hits=positive_objects=positive_preds=0;negative_preds=0;negative_high=0;frames=[]
        for item in protocol['selected']:
            row=item['row'];truth=row['truth']['objects']
            with Image.open(row['rgb_path']) as im:r=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
            used=set()
            for box,cls in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist()):
                vals=[(iou(box,o['bbox_xyxy']),i) for i,o in enumerate(truth) if i not in used and o['class_name']==model.names[int(cls)]]
                overlap,index=max(vals,default=(0,-1))
                if overlap>=.5:used.add(index)
            positive_hits+=len(used);positive_objects+=len(truth);positive_preds+=len(r.boxes)
        for item in ab_protocol['selected']:
            if item['kind']!='accepted_background_crop':continue
            with Image.open(item['path']) as im:r=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
            negative_preds+=len(r.boxes);negative_high+=sum(float(c)>=.5 for c in r.boxes.conf.tolist())
        results[name]={'weights_sha256':file_sha256(path),'positive_hits':positive_hits,'positive_objects':positive_objects,'positive_predictions':positive_preds,'negative_crop_predictions':negative_preds,'negative_crop_predictions_conf_ge_0_5':negative_high}
    report={'results':results,'confidence':.37,'matching_iou':.5,'training_set_only':True,'promotable':False,
            'inputs':{str(p):file_sha256(p) for p in [mem/'protocol.json',ab/'protocol.json',*paths.values(),Path(__file__)]},
            'limits':['Both models are evaluated on data used for training.','Negative crops are derived from prior model predictions and are not independent evidence.','No deployment or formal qualification conclusion.']}
    report['identity']=object_sha256(report);write_json(ab/'comparison.json',report)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
