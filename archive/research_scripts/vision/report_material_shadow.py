"""Summarize measured integration behavior, not flight certification."""
from collections import defaultdict
from pathlib import Path
import json
import subprocess
import sys
from scripts.vision import material_shadow as s
from scripts.vision.evaluate_paired_visual_factors import match
from src.vision.replay.static_runtime import timing_summary
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1').resolve()

def main():
    p=ROOT/'assessment-001/comparison.json';a=s.reference.checked(p)
    cp=s.reference.OUT/'evaluation-v1/units/material-routed-480-7/completion.json';c=s.reference.checked(cp);old=s.reference.checked(c['result'])
    by={r['image_sha256']:r for r in old['rows']+old['negative_rows']};results={}
    for name,data in a['results'].items():
        stats=defaultdict(lambda:dict(frames=0,truth=0,matched=0,planned_hits=0,predictions=0))
        counts=0;maxbox=maxconfidence=0.;negative_fp=0;assignment_differences=0
        for row in data['rows']:
            ref=by[row['image_sha256']];pred=row['predictions'];prior=ref['predictions']
            if len(pred)!=len(prior) or [x['class_name'] for x in pred]!=[x['class_name'] for x in prior]:counts+=1
            else:
                for x,y in zip(pred,prior):
                    maxbox=max(maxbox,max(abs(j-k) for j,k in zip(x['bbox_xyxy'],y['bbox_xyxy'])))
                    maxconfidence=max(maxconfidence,abs(x['confidence']-y['confidence']))
            if 'truth' not in ref:negative_fp+=bool(pred);continue
            matches,_,used=match(pred,ref['truth']);st=stats[ref['variant']]
            st['frames']+=1;st['truth']+=len(ref['truth']);st['matched']+=len(matches);st['predictions']+=len(pred);st['planned_hits']+=ref['planned_truth_index'] in used
            assignment_differences+=used!={m['truth_index'] for m in ref['matches']}
        for st in stats.values():st.update(recall=st['matched']/st['truth'],planned_hit=st['planned_hits']/st['frames'],precision=st['matched']/st['predictions'])
        results[name]=dict(summary=dict(stats),negative_fp_frames=negative_fp,negative_frames=48,fpr=negative_fp/48,count_or_class_order_changed_frames=counts,matched_truth_set_changed_frames=assignment_differences,max_coordinate_delta_same_order_px=maxbox,max_confidence_delta_same_order=maxconfidence,exact_match_frames=data['exact_match_count'],warm_calls=data['warm_calls'])
    live=ROOT/'isolated-probe-001/live';completion=s.reference.checked(live/'completion.json');probe=s.reference.checked(live.parent/'probe.json')
    rows=[json.loads(x) for x in (live/'detections.jsonl').read_text().splitlines()]
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_material_shadow','tests.test_gazebo_camera_pipeline','tests.test_camera_decoder','-q'],capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stderr)
    paths=[p,cp,Path(c['result']),live/'completion.json',live/'detections.jsonl',live.parent/'probe.json',Path(__file__).resolve(),Path('tests/test_material_shadow.py').resolve()]
    r=dict(status='integration_assessment_complete_preprocessing_issue_identified',results=results,live=dict(counts=completion['counts'],inference_cold_ms=rows[0]['inference_call_ms'],inference_warm=timing_summary([r['inference_call_ms'] for r in rows[1:]]),receive_age_ms=timing_summary([r['receive_age_at_inference_s']*1000 for r in rows]),all_frames_prediction_count_two=all(len(r['predictions'])==2 for r in rows),requested_duration_s=15,owned_processes_exited=probe['owned_processes_exited']),tests=tests.stdout+tests.stderr,review=dict(nature='AI辅助审核',scope='first live frame only',observation='右侧三柱顶部及青色大主体、中央受前景遮挡的块体可见；两个预测分别落在右侧设备和中央块体。未逐帧认证实例归属、完整标签或像素可见性。'),limits=['Single static known-development pose, seed7; not independent scene or flight evidence.','Receive age excludes unknown transport backlog; no capture-to-decision latency claim.','Current live run uses rect=True; proposed rect=False tested offline only.','No production config changed; no training or promotion.'],training_admitted=False,promotable=False,inputs={str(q):file_sha256(q) for q in paths})
    write_record(ROOT/'assessment-001/report.json',r)
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
