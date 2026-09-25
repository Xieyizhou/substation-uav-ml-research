"""Verify completed control capture receipts and raw artifacts."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,pose_close


def main():
    base=ROOT/'data/research/ml_training_recovery_v1/control-matrix-capture-v1'
    progress=json.loads((base/'progress.json').read_text());rows=[];inputs={}
    for run in progress['runs']:
        plan=read_record(run['plan_path']);receipt=read_record(run['receipt_path'])
        assert receipt['plan_identity']==plan['identity']
        assert receipt['status']=='complete_pending_review'
        views={v['view_id']:v for v in plan['calibration_views']}
        assert len(receipt['views'])==len(views)
        for p in (run['plan_path'],run['receipt_path']):inputs[p]=file_sha256(Path(p))
        for row in receipt['views']:
            assert row['status']=='captured' and pose_close(row['actual_pose'],views[row['view_id']])
            for key,hashkey in [('rgb_path','image_sha256'),('depth_path','depth_sha256')]:
                inputs[row[key]]=file_sha256(Path(row[key]));assert inputs[row[key]]==row[hashkey]
            assert row['training_admitted'] is False
            rows.append({'view_id':row['view_id'],'batch':run['name'],
                         'expected_category':row['expected_category'],
                         'expected_class_present':row['expected_category'] in {o['class_name'] for o in row['truth']['objects']},
                         'rgb_path':row['rgb_path']})
    report={'inputs':inputs,'batches':len(progress['runs']),'captured':len(rows),
            'complete_80_views':len(rows)==80,'views':rows,'training_admitted':False,
            'limits':['Artifact and pose verification only; pixel occlusion and diagnostic regions remain unreviewed.',
                      'Class presence is not instance presence; cabinet is outside target taxonomy.']}
    report['identity']=object_sha256(report);write_json(base/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','views')},indent=2))


if __name__=='__main__':main()
