"""Evaluate diagnostic weights on the alternate simple world package."""
import json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou

def main():
 from ultralytics import YOLO
 base=ROOT/'data/research/ml_training_recovery_v1/simple-independent-scene-v1';manifest=json.loads((base/'manifest.json').read_text());plan=json.loads(Path(manifest['plan_path']).read_text());receipt=json.loads(Path(manifest['receipt_path']).read_text());views={v['view_id']:v for v in plan['calibration_views']};rows=[]
 inputs={str(p):file_sha256(p) for p in [base/'manifest.json',base/'plan/plan.json',base/'capture/collection-receipt.json',Path(__file__)]}
 for row in receipt['views']:
  p=Path(row['rgb_path']);inputs[str(p)]=file_sha256(p);v=views[row['view_id']];rows.append({'view_id':row['view_id'],'path':str(p),'category':v['category'],'truth':row['truth']['objects']})
 paths={'positive_only':ROOT/'data/research/ml_training_recovery_v1/memorization-v1/fit-batchmatched/weights/last.pt','negative_augmented':ROOT/'data/research/ml_training_recovery_v1/negative-ab-v1/fit/weights/last.pt'};results={}
 for name,weight in paths.items():
  model=YOLO(str(weight));scored=[]
  for row in rows:
   with Image.open(row['path']) as im:r=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
   preds=[{'bbox_xyxy':b,'class_name':model.names[int(k)],'confidence':c} for b,k,c in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist(),r.boxes.conf.tolist())];truth=row['truth'];used=set();hits=0
   for p in preds:
    vals=[(iou(p['bbox_xyxy'],o['bbox_xyxy']),i) for i,o in enumerate(truth) if i not in used and o['class_name']==p['class_name']];overlap,index=max(vals,default=(0,-1))
    if overlap>=.5:used.add(index);hits+=1
   expected_present=any(o['class_name']==row['category'] for o in truth)
   scored.append({**row,'truth_count':len(truth),'hits':hits,'predictions':len(preds),'high_confidence_predictions':sum(float(p['confidence'])>=.5 for p in preds),'expected_class_present':expected_present,'expected_hit':expected_present and any(p['class_name']==row['category'] and max((iou(p['bbox_xyxy'],o['bbox_xyxy']) for o in truth if o['class_name']==row['category']),default=0)>=.5 for p in preds)})
  results[name]={'weights_sha256':file_sha256(weight),'rows':scored}
 summary={}
 for n,r in results.items():summary[n]={'frames':len(r['rows']),'target_frames':sum(x['truth_count']>0 for x in r['rows']),'target_hits':sum(x['hits'] for x in r['rows']),'target_objects':sum(x['truth_count'] for x in r['rows']),'expected_class_hits':sum(x['expected_class_present'] and x['expected_hit'] for x in r['rows']),'observable_expected_frames':sum(x['expected_class_present'] for x in r['rows']),'expected_absent_frames':sum(not x['expected_class_present'] for x in r['rows']),'no_target_frames':sum(x['truth_count']==0 for x in r['rows']),'no_target_frames_with_predictions':sum(x['truth_count']==0 and x['predictions']>0 for x in r['rows']),'predictions':sum(x['predictions'] for x in r['rows']),'high_confidence_predictions':sum(x['high_confidence_predictions'] for x in r['rows']),'by_expected_category':{c:{'frames':sum(x['category']==c for x in r['rows']),'observable_frames':sum(x['category']==c and x['expected_class_present'] for x in r['rows']),'expected_absent_frames':sum(x['category']==c and not x['expected_class_present'] for x in r['rows']),'expected_class_hits':sum(x['category']==c and x['expected_class_present'] and x['expected_hit'] for x in r['rows']),'matched_hits':sum(x['hits'] for x in r['rows'] if x['category']==c),'truth_objects':sum(x['truth_count'] for x in r['rows'] if x['category']==c),'predictions':sum(x['predictions'] for x in r['rows'] if x['category']==c)} for c in ('cabinet','empty_ground','switchgear','transformer')}}
 report={'results':results,'summary':summary,'inputs':inputs,'confidence':.37,'matching_iou':.5,'status':'simple_independent_scene_complete','training_admitted':False,'promotable':False,'limits':['This is one alternate simple world package with 12 views.','No threshold selection or formal validation.','Expected-class recall is counted only when the expected class is present in full_2d truth; expected-absent rows are scene/visibility or annotation holds.','Cabinet and empty-ground frames have no expected target boxes in this package; only no-target prediction presence is reported.']};report['identity']=object_sha256(report);write_json(base/'evaluation.json',report);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
