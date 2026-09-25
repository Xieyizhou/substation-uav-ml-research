"""Verify actual corrected sidecar against every historical development frame."""
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scripts.vision import material_shadow as s
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from scripts.vision.evaluate_paired_visual_factors import match
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=Path('data/research/material-shadow-v1/rect-fixed-offline-v1').resolve()

def main():
    OUT.mkdir(parents=True,exist_ok=False);identity=s.model_identity(7)
    if s.PROTOCOL.get('rect') is not False:raise ValueError('Missing explicit square padding')
    p,n,deps=_load_inputs();cp=s.reference.OUT/'evaluation-v1/units/material-routed-480-7/completion.json';c=s.reference.checked(cp);old=s.reference.checked(c['result'])
    by={r['image_sha256']:r for r in old['rows']+old['negative_rows']};rows=[]
    with s.locked_threads(4):
        m=YOLO(identity['weights'])
        for row in [r for r,_ in p]+n:
            out=s.infer(m,np.array(Image.open(row['image_path']).convert('RGB')))
            pred=[dict(class_name=x['class_name'],confidence=x['confidence'],bbox_xyxy=x['xyxy']) for x in out['predictions']];ref=by[row['image_sha256']];prior=ref['predictions']
            if len(pred)!=len(prior) or [x['class_name'] for x in pred]!=[x['class_name'] for x in prior]:raise ValueError('Count or classes differ')
            dx=max((abs(a-b) for x,y in zip(pred,prior) for a,b in zip(x['bbox_xyxy'],y['bbox_xyxy'])),default=0.)
            dc=max((abs(x['confidence']-y['confidence']) for x,y in zip(pred,prior)),default=0.)
            if dx>.001 or dc>.00001:raise ValueError('Numerical tolerance exceeded')
            if 'truth' in ref:
                _,_,used=match(pred,ref['truth'])
                if used!={r['truth_index'] for r in ref['matches']}:raise ValueError('Matched truth differs')
            rows.append(dict(image_sha256=row['image_sha256'],predictions=pred,coordinate_delta_px=dx,confidence_delta=dc,inference_call_ms=out['inference_call_ms']))
    for q in (cp,Path(c['result']),Path(__file__),Path(s.__file__),Path(identity['weights'])):deps[str(q.resolve())]=file_sha256(q)
    write_record(OUT/'completion.json',dict(status='all_96_corrected_runtime_predictions_verified',model=identity,rows=rows,comparison_tolerances=dict(coordinates_px=.001,confidence=.00001),training_admitted=False,promotable=False,inputs=deps))
    print('96/96 prediction and matched-instance checks passed',flush=True)

if __name__=='__main__':main()
