"""Read-only integration assessment; compare current and proposed rect=False."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scripts.vision import material_shadow as s
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from src.vision.replay.static_runtime import timing_summary
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=Path('data/research/material-shadow-v1/assessment-001').resolve()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    identity=s.model_identity(7);paired,negative,deps=_load_inputs()
    cp=s.reference.OUT/'evaluation-v1/units/material-routed-480-7/completion.json'
    c=s.reference.checked(cp);old=s.reference.checked(c['result'])
    by={r['image_sha256']:r for r in old['rows']+old['negative_rows']}
    sources=[r for r,_ in paired]+negative
    results={}
    with s.locked_threads(4):
        for name,rect in [('current_rect_true',True),('proposed_rect_false',False)]:
            model=YOLO(identity['weights']);rows=[]
            with patch.dict(s.PROTOCOL,{'rect':rect}):
                for row in sources:
                    rgb=np.array(Image.open(row['image_path']).convert('RGB'))
                    out=s.infer(model,rgb)
                    predictions=[dict(class_name=x['class_name'],confidence=x['confidence'],bbox_xyxy=x['xyxy']) for x in out['predictions']]
                    prior=by[row['image_sha256']]['predictions']
                    rows.append(dict(image=row['image_path'],image_sha256=row['image_sha256'],predictions=predictions,exact_historical_match=predictions==prior,inference_call_ms=out['inference_call_ms']))
            results[name]=dict(frame_count=len(rows),exact_match_count=sum(r['exact_historical_match'] for r in rows),cold_call_ms=rows[0]['inference_call_ms'],warm_calls=timing_summary([r['inference_call_ms'] for r in rows[1:]]),rows=rows)
            print(name,results[name]['exact_match_count'],'/',len(rows),flush=True)
    for p in (cp,Path(c['result']),Path(s.__file__),Path(__file__),Path(identity['weights'])):deps[str(p.resolve())]=file_sha256(p)
    write_record(OUT/'comparison.json',dict(status='assessment_complete_no_production_config_change',model=identity,results=results,historical_seed7_summary=old['summary'],historical_seed7_negative_summary=old['negative_summary'],training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':main()
