"""Verify warmup ordering, equivalence and measured static latency."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.vision import material_shadow as s
from src.vision.replay.static_runtime import timing_summary
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1').resolve()

def main():
    base=ROOT/'prewarm-retest-v1';paths=[base/'completion.json',ROOT/'prewarm-offline-v1/completion.json']
    for p in paths:s.reference.checked(p)
    rows=[]
    for i in range(1,4):
        p=base/f'live-{i:02}/live';c=s.reference.checked(p/'completion.json');w=s.reference.checked(p/'warmup.json')
        log=[json.loads(x) for x in (p/'detections.jsonl').read_text().splitlines()]
        if len(log)!=c['counts']['inferred'] or any(r['frame']['receive_monotonic_timestamp']<w['finished_monotonic'] for r in log):raise ValueError('Warmup not before subscription')
        rows.append(dict(run=i,counts=c['counts'],warmup_ms=w['duration_ms'],first_live_inference_ms=log[0]['inference_call_ms'],steady={k:timing_summary([r[k] for r in log[1:]]) for k in ('inference_call_ms','receive_to_result_ms','source_encode_dispatch_ms','application_queue_wait_ms','decode_ms')}))
        paths.extend([p/'completion.json',p/'warmup.json',p/'detections.jsonl'])
    oldpath=ROOT/'rect-fixed-retest-v1/latency-report.json';old=json.loads(oldpath.read_text())
    archive=ROOT/'rect-fixed-retest-v1/material_shadow_before_warmup.py'
    if file_sha256(archive)!=old['inputs'][str(Path(s.__file__).resolve())]:raise ValueError('Old version snapshot mismatch')
    t=subprocess.run([sys.executable,'-m','unittest','tests.test_material_shadow','tests.test_gazebo_camera_pipeline','tests.test_camera_decoder','-q'],capture_output=True,text=True)
    if t.returncode:raise RuntimeError(t.stderr)
    paths.extend([oldpath,archive,Path(__file__).resolve(),Path('scripts/vision/verify_material_shadow_warmup.py').resolve(),Path('scripts/vision/retest_material_shadow_warmup.py').resolve(),Path('tests/test_material_shadow.py').resolve()])
    write_record(base/'comparison.json',dict(status='warmup_retest_complete',runs=rows,baseline_runs=old['runs'],offline_frames_verified=96,test_output=t.stdout+t.stderr,limits=['Historical baseline second run overlapped offline inference; compare descriptively, not causal statistical significance.','Same single static known-development pose, fixed seed7; no PX4 or flight.','Warmup moves startup cost before subscription; it does not remove startup time or unknown pre-receive backlog.','PNG encode/write/read/decode path inspected, not optimized in this experiment.'],training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
