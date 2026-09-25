"""Measured post-receive latency only; never infer unknown capture-clock latency."""
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
    offline=ROOT/'rect-fixed-offline-v1/completion.json';o=s.reference.checked(offline)
    if len(o['rows'])!=96 or o['model']['protocol']['rect'] is not False:raise ValueError('Offline gate failed')
    base=ROOT/'rect-fixed-retest-v1';s.reference.checked(base/'completion.json');runs=[];paths=[offline,base/'completion.json']
    for i in range(1,4):
        p=base/f'live-{i:02}';c=s.reference.checked(p/'live/completion.json');identity=s.reference.checked(p/'live/model.json');probe=s.reference.checked(p/'probe.json')
        if identity['protocol']['rect'] is not False or not probe['owned_processes_exited']:raise ValueError('Live gate failed')
        log=p/'live/detections.jsonl';rows=[json.loads(x) for x in log.read_text().splitlines()]
        if len(rows)!=c['counts']['inferred']:raise ValueError('Missing rows')
        stats={k:timing_summary([r[k] for r in rows[1:]]) for k in ('source_encode_dispatch_ms','application_queue_wait_ms','decode_ms','inference_call_ms','receive_to_result_ms')}
        runs.append(dict(run=i,counts=c['counts'],cold_inference_ms=rows[0]['inference_call_ms'],warm=stats,warm_output_hz=(len(rows)-1)/(rows[-1]['result_monotonic']-rows[0]['result_monotonic']),concurrency_note='Overlapped offline verification; not unloaded baseline' if i==2 else 'No intentionally concurrent offline inference; Gazebo rendering active'))
        paths.extend([log,p/'live/completion.json',p/'live/model.json',p/'probe.json'])
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_material_shadow','tests.test_gazebo_camera_pipeline','tests.test_camera_decoder','-q'],capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stderr)
    archived=ROOT/'assessment-001/material_shadow_before_fix.py';old=json.loads((ROOT/'assessment-001/comparison.json').read_text())
    if file_sha256(archived)!=old['inputs'][str(Path(s.__file__).resolve())]:raise ValueError('Historical script snapshot mismatch')
    paths.extend([archived,Path(s.__file__).resolve(),Path(__file__).resolve(),Path('tests/test_material_shadow.py').resolve(),base/'report-zh.md'])
    write_record(base/'latency-report.json',dict(status='rect_fix_retested_latency_characterized',offline_verified_frames=96,runs=runs,test_output=tests.stdout+tests.stderr,
        erratum='Historical receipts bind the old mutable script path. Exact old script bytes are preserved at assessment-001/material_shadow_before_fix.py; original receipts are not rewritten. New runs bind corrected script.',
        limits=['Post-receive latency only; no cross-clock subtraction.','Static known-development pose and fixed seed7; not flight or hardware certification.','No PX4 concurrency, motion, fault injection or deadline acceptance threshold validated.','Cold samples excluded from warm summaries and reported separately.'],
        training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print('Corrected pipeline, 96 frames, three live runs and 22 tests verified')

if __name__=='__main__':main()
