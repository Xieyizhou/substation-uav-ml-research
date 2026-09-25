"""Report observed SITL telemetry; not motor-loop timing or flight approval."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.vision import material_shadow as s
from scripts.flight.check_px4_shadow import PX4,BUILD
from src.vision.replay.static_runtime import timing_summary
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1/px4-unarmed-concurrency-v1/attempt-002').resolve()

def main():
    receipt=s.reference.checked(ROOT/'receipt.json');t=s.reference.checked(ROOT/'telemetry.json');samples=t['samples']
    if receipt['status']!='unarmed_sitl_concurrency_complete' or not receipt['owned_processes_exited']:raise ValueError('Incomplete SITL test')
    if any(r['value'] is not False for r in samples if r['stream'] in ('armed','in_air')):raise ValueError('Unexpected arm/air state')
    stats={}
    for ph in ('baseline','vision','recovery'):
        stats[ph]={}
        for name in ('position','attitude'):
            times=[x['monotonic'] for x in samples if x['phase']==ph and x['stream']==name]
            gaps=[(b-a)*1000 for a,b in zip(times,times[1:])]
            stats[ph][name]=dict(samples=len(times),rate_hz=(len(times)-1)/(times[-1]-times[0]),interval_ms=timing_summary(gaps))
    s.reference.checked(ROOT/'vision/latest-implementation.json');v=s.reference.checked(ROOT/'vision/completion.json')
    rows=[json.loads(x) for x in (ROOT/'vision/detections.jsonl').read_text().splitlines()]
    vision=timing_summary([r['receive_to_logged_result_ms'] for r in rows[1:]])
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_px4_shadow_check','tests.test_gazebo_camera_memory','tests.test_material_shadow','tests.test_gazebo_camera_pipeline','tests.test_camera_decoder','-q'],capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stderr)
    paths=[ROOT/'receipt.json',ROOT/'telemetry.json',ROOT/'px4.log',ROOT/'gazebo.log',ROOT/'vision.log',ROOT/'vision/completion.json',ROOT/'vision/latest-implementation.json',Path(__file__).resolve(),Path('tests/test_px4_shadow_check.py').resolve(),PX4/'Tools/simulation/gz/server.config',BUILD/'etc/init.d-posix/rcS',BUILD/'etc/init.d-posix/px4-rc.gzsim',BUILD/'etc/init.d-posix/px4-rc.mavlink']
    write_record(ROOT/'analysis.json',dict(status='unarmed_concurrency_checked_not_flight_certified',telemetry=stats,vision_counts=v['counts'],vision_post_receive_latency_ms=vision,last_health=[r for r in samples if r['stream']=='health'][-1],tests=tests.stdout+tests.stderr,training_admitted=False,promotable=False,limits=['MAVSDK callback intervals, not PX4 control-loop periods.','Vision phase includes process startup/prewarm; actual image sampling is20s.','One unarmed SITL run; no flight, failsafe injection or physical hardware.','No prospective timing acceptance threshold; no safety-pass claim.'],inputs={str(p):file_sha256(p) for p in paths}))
    print(stats);print(vision)

if __name__=='__main__':main()
