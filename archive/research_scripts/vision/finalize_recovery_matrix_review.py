"""Bind assistant visual observations; keep uncertain ROI attribution separate."""
import json
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json


def main():
    source=ROOT/'data/research/ml_training_recovery_v1/matrix-baseline-v1/report.json'
    report=json.loads(source.read_text())
    # Recorded after inspecting all ten sheets, not inferred from occupancy rays.
    hidden={3,4,31}
    uncertain={5}
    partial={1,2,30,64,65,66,69,77,79}
    cabinet_fp={33,38,44,48}
    cabinet_ambiguous={10,20,23,34,39,40,68}
    rows=[];counts=defaultdict(lambda:{'frames':0,'eligible_instance_truth':0,'correct_hits':0})
    for frame in report['frames']:
        n=frame['index'];category=frame['view']['category']
        visibility='hidden' if n in hidden else 'uncertain' if n in uncertain else 'partial' if n in partial else 'clear'
        row={'index':n,'view_id':frame['view']['view_id'],'category':category,
             'visual_visibility':visibility,'geometric_proxy':frame['view']['occlusion_proxy'],
             'region_status':'coarse_occupancy_region_not_label',
             'cabinet_attribution':'confirmed_target_class_false_positive' if n in cabinet_fp else 'ambiguous_overlap' if n in cabinet_ambiguous else 'no_confirmed_cabinet_attribution',
             'correct_instance_hit':frame['correct_instance_hit'], 'target_truth_count':len(frame['target_truth'])}
        rows.append(row)
        if category=='switchgear':
            item=counts[visibility];item['frames']+=1
            if visibility in ('clear','partial') and len(frame['target_truth'])==1:
                item['eligible_instance_truth']+=1;item['correct_hits']+=int(frame['correct_instance_hit'])
    result={'inputs':{str(source):file_sha256(source),__file__:file_sha256(Path(__file__))},
            'visual_evidence':report['visuals'],'reviewer':'assistant',
            'method':'All 80 development images reviewed at 640x360 with instance truth, occupancy ROI and predictions overlaid. Not independent human annotation; no pixel masks.',
            'observations':rows,'switchgear_by_visual_visibility':dict(counts),
            'cabinet_frames':sum(r['category']=='cabinet' for r in rows),
            'confirmed_cabinet_false_positive_indices':sorted(cabinet_fp),
            'ambiguous_cabinet_overlap_indices':sorted(cabinet_ambiguous),
            'training_admitted':False,
            'limits':['Visual review saw predictions; visibility observations are not blinded.',
                      'Uncertain and hidden instances excluded from visible recall denominator.',
                      'Cabinet counts describe reviewed attribution, not whole-frame FPR.',
                      'Coarse occupancy ROI remains unsuitable for automatic cabinet IoU scoring.',
                      'Same scenes and correlated views; not an independent qualification set.']}
    result['identity']=object_sha256(result);write_json(source.parent/'visual-review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','visual_evidence','observations')},indent=2))


if __name__=='__main__':main()
