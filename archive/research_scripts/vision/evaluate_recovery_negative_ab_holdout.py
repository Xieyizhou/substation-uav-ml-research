"""Independent development recheck for the negative-sample diagnostic."""
import json
import sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou,indexed


def score(model,path,truth,expected=None):
    with Image.open(path) as im:r=model.predict(im.convert('RGB'),imgsz=640,conf=.37,iou=.7,rect=False,device='cpu',verbose=False,agnostic_nms=False,max_det=300)[0]
    preds=[{'bbox_xyxy':b,'class_name':model.names[int(k)],'confidence':c} for b,k,c in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist(),r.boxes.conf.tolist())]
    used=set();hits=0
    for p in preds:
        vals=[(iou(p['bbox_xyxy'],o['bbox_xyxy']),i) for i,o in enumerate(truth) if i not in used and o['class_name']==p['class_name']]
        overlap,index=max(vals,default=(0,-1))
        if overlap>=.5:used.add(index);hits+=1
    expected_present=None;expected_hit=None
    if expected:
        expected_present=any(o['class_name']==expected for o in truth)
        expected_hit=expected_present and any(p['class_name']==expected and max((iou(p['bbox_xyxy'],o['bbox_xyxy']) for o in truth if o['class_name']==expected),default=0)>=.5 for p in preds)
    return {'predictions':len(preds),'matched_hits':hits,'truth_count':len(truth),'expected_class':expected,'expected_class_present':expected_present,'expected_class_hit':expected_hit,
            'high_confidence_predictions':sum(float(p['confidence'])>=.5 for p in preds)}


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1';out=base/'negative-ab-v1'
    paths={'positive_only':out.parent/'memorization-v1/fit-batchmatched/weights/last.pt','negative_augmented':out/'fit/weights/last.pt'}
    manifest=base/'paired-calibration-v1/plan-manifest.json';matrix_progress=base/'control-matrix-capture-v1/progress.json'
    inputs={str(p):file_sha256(p) for p in [manifest,matrix_progress,*paths.values(),Path(__file__)]}
    datasets={'held_targets':[],'matrix_controls':[]}
    m=json.loads(manifest.read_text())
    for run in m['runs']:
        if run['mode']!='full_2d':continue
        receipt=json.loads((base/'paired-calibration-v1'/run['name']/'capture/collection-receipt.json').read_text())
        for row in receipt['views']:
            if row['view_id'] not in run['held_target_view_ids']:continue
            datasets['held_targets'].append({'path':row['rgb_path'],'truth':row['truth']['objects'],'expected':row['expected_category'],'view_id':row['view_id'],'map_id':row['map_id']})
    for run in json.loads(matrix_progress.read_text())['runs']:
        receipt=json.loads(Path(run['receipt_path']).read_text());plan=json.loads(Path(run['plan_path']).read_text());views={v['view_id']:v for v in plan['calibration_views']}
        for row in receipt['views']:
            view=views[row['view_id']];datasets['matrix_controls'].append({'path':row['rgb_path'],'truth':row['truth']['objects'],'expected':row['expected_category'],'view_id':row['view_id'],'map_id':row['map_id']})
    assert len(datasets['held_targets'])==19 and len(datasets['matrix_controls'])==80
    results={}
    for name,weight in paths.items():
        model=YOLO(str(weight));groups={}
        for group,rows in datasets.items():
            scored=[]
            for row in rows:
                p=Path(row['path']);inputs[str(p)]=file_sha256(p)
                scored.append({**row,'score':score(model,p,row['truth'],row['expected'])})
            groups[group]=scored
        results[name]={'weights_sha256':file_sha256(weight),'groups':groups}
    summary={}
    for name,res in results.items():
        summary[name]={}
        for group,rows in res['groups'].items():
            expected=[r for r in rows if r['score']['expected_class'] in ('transformer','switchgear','capacitor_bank','reactor')]
            summary[name][group]={'frames':len(rows),'truth_objects':sum(r['score']['truth_count'] for r in rows),
                                  'matched_hits':sum(r['score']['matched_hits'] for r in rows),
                                  'predictions':sum(r['score']['predictions'] for r in rows),
                                  'expected_class_hits':sum(bool(r['score']['expected_class_present'] and r['score']['expected_class_hit']) for r in expected),
                                  'expected_class_frames':sum(bool(r['score']['expected_class_present']) for r in expected),
                                  'expected_class_absent_frames':sum(not r['score']['expected_class_present'] for r in expected),
                                  'high_confidence_predictions':sum(r['score']['high_confidence_predictions'] for r in rows)}
    report={'results':results,'summary':summary,'confidence':.37,'matching_iou':.5,'inputs':inputs,'status':'independent_development_recheck_complete','training_admitted':False,'promotable':False,
            'limits':['The 19 held views share source maps with earlier collection and are development diagnostics, not protected validation.','The 80 controls are correlated views from three maps.','Expected-class recall is counted only when the expected class is present in full_2d truth; expected-absent rows are scene/visibility or annotation holds.','Cabinet has no positive target label; predictions on cabinet rows are not a formal FPR.','No threshold selection or model promotion performed.']}
    report['identity']=object_sha256(report);write_json(out/'independent-recheck.json',report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
