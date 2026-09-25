"""Evaluate both diagnostic weights on alternate-world full_2d captures."""
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
    base=ROOT/'data/research/ml_training_recovery_v1';cross=base/'cross-scene-recheck-v1';ab=base/'negative-ab-v1';mem=base/'memorization-v1'
    progress=json.loads((cross/'progress.json').read_text())
    paths={'positive_only':mem/'fit-batchmatched/weights/last.pt','negative_augmented':ab/'fit/weights/last.pt'}
    inputs={str(p):file_sha256(p) for p in [cross/'manifest.json',cross/'progress.json',*paths.values(),Path(__file__)]}
    known_hashes=set()
    training_protocol=mem/'protocol.json';inputs[str(training_protocol)]=file_sha256(training_protocol)
    for item in json.loads(training_protocol.read_text())['selected']:known_hashes.add(item['row']['image_sha256'])
    for root in (base/'paired-calibration-v1',base/'control-matrix-capture-v1'):
        for receipt_path in root.rglob('collection-receipt.json'):
            receipt=json.loads(receipt_path.read_text());inputs[str(receipt_path)]=file_sha256(receipt_path)
            for item in receipt.get('views',[]):
                if item.get('status')=='captured':known_hashes.add(item['image_sha256'])
    rows=[]
    for run in progress['runs']:
        plan=json.loads(Path(run['plan_path']).read_text());receipt=json.loads(Path(run['receipt_path']).read_text());views={v['view_id']:v for v in plan['calibration_views']}
        for row in receipt['views']:
            v=views[row['view_id']];p=Path(row['rgb_path']);inputs[str(p)]=file_sha256(p)
            image_hash=file_sha256(p)
            rows.append({'map_id':row['map_id'],'view_id':row['view_id'],'path':str(p),'image_sha256':image_hash,'prior_diagnostic_pixel_overlap':image_hash in known_hashes,'truth':row['truth']['objects'],'expected':row['expected_category'],'world_sha256':plan['files']['world.sdf']})
    assert len(rows)==27 and len({r['world_sha256'] for r in rows})==3
    results={}
    for name,weight in paths.items():
        model=YOLO(str(weight));scored=[]
        for row in rows:
            with Image.open(row['path']) as im:r=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
            preds=[{'bbox_xyxy':b,'class_name':model.names[int(k)],'confidence':c} for b,k,c in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist(),r.boxes.conf.tolist())]
            truth=row['truth'];used=set();hits=0
            for p in preds:
                vals=[(iou(p['bbox_xyxy'],o['bbox_xyxy']),i) for i,o in enumerate(truth) if i not in used and o['class_name']==p['class_name']]
                overlap,index=max(vals,default=(0,-1))
                if overlap>=.5:used.add(index);hits+=1
            expected_present=any(o['class_name']==row['expected'] for o in truth)
            expected_hit=expected_present and any(p['class_name']==row['expected'] and max((iou(p['bbox_xyxy'],o['bbox_xyxy']) for o in truth if o['class_name']==row['expected']),default=0)>=.5 for p in preds)
            scored.append({**row,'hits':hits,'truth_count':len(truth),'predictions':len(preds),'expected_class_present':expected_present,'expected_hit':expected_hit,'high_confidence_predictions':sum(float(p['confidence'])>=.5 for p in preds)})
        results[name]={'weights_sha256':file_sha256(weight),'rows':scored}
    summary={}
    for name,res in results.items():
            independent=[r for r in res['rows'] if not r['prior_diagnostic_pixel_overlap']]
            summary[name]={'frames':len(independent),'overlap_frames':sum(r['prior_diagnostic_pixel_overlap'] for r in res['rows']),'truth_objects':sum(r['truth_count'] for r in independent),'matched_hits':sum(r['hits'] for r in independent),'predictions':sum(r['predictions'] for r in independent),'expected_class_hits':sum(r['expected_class_present'] and r['expected_hit'] for r in independent),'observable_expected_frames':sum(r['expected_class_present'] for r in independent),'expected_absent_frames':sum(not r['expected_class_present'] for r in independent),'high_confidence_predictions':sum(r['high_confidence_predictions'] for r in independent),'by_map':{m:{'frames':sum(r['map_id']==m for r in independent),'hits':sum(r['hits'] for r in independent if r['map_id']==m),'objects':sum(r['truth_count'] for r in independent if r['map_id']==m),'predictions':sum(r['predictions'] for r in independent if r['map_id']==m),'observable_expected_frames':sum(r['map_id']==m and r['expected_class_present'] for r in independent),'expected_class_hits':sum(r['map_id']==m and r['expected_class_present'] and r['expected_hit'] for r in independent)} for m in ('complex','medium','simple')}}
    report={'results':results,'summary':summary,'inputs':inputs,'confidence':.37,'matching_iou':.5,'status':'cross_scene_recheck_complete','training_admitted':False,'promotable':False,
            'limits':['Alternate world snapshots are development diagnostics, not protected validation.','The three maps are still simulated and the view counts are small.','Expected-class recall is counted only when the expected class is present in full_2d truth; expected-absent rows are scene/visibility or annotation holds.','No threshold selection, deployment claim, or model promotion.']}
    report['identity']=object_sha256(report);write_json(cross/'evaluation.json',report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
