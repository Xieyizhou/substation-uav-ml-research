"""Four saved-frame occlusion replays; no training or historical edits."""
import argparse
import asyncio
import os
import shutil
from pathlib import Path
from scripts.vision.audit_redistribution_targets import OUT as PRIOR, OCCLUDED, read, verify, frozen, file_sha256, ROOT, replay, instance_mapping

OUT = PRIOR / 'same-frame-occlusion-replay-v1'

def freeze():
    path = OUT / 'protocol.json'
    if path.exists():
        verify(read(path))
        return
    review = read(PRIOR / 'review.json'); verify(review)
    frames = []; paths = [PRIOR / 'review.json', PRIOR / 'completion.json', Path(__file__), Path(replay.__file__)]
    for d in review['decisions']:
        if d['event_id'] not in OCCLUDED: continue
        pp = Path(d['source_world']).parent / 'plan.json'
        plan = read(pp); mapping = instance_mapping(plan)
        if mapping[d['runtime_label']]['object_id'] != d['object_id']: raise ValueError('Mapping changed')
        frames.append(dict(event_id=d['event_id'], member_id=d['member_id'], review_ids=[d['event_id']],
            source_image=d['source_image'], source_world=d['source_world'], source_plan=str(pp),
            source_receipt=d['source_receipt'], actual_pose=d['actual_pose'], world_name=plan['world_name'],
            instance_mapping={str(k): v for k,v in mapping.items()},
            events=[dict(review_id=d['event_id'], runtime_label=d['runtime_label'], object_id=d['object_id'], bbox_xyxy=d['truth']['bbox_xyxy'])]))
        paths.extend([pp, pp.parent/'obstacles.json', Path(d['source_world']), Path(d['source_image']), Path(d['source_receipt'])])
    if {f['event_id'] for f in frames} != OCCLUDED: raise ValueError('Scope changed')
    OUT.mkdir(exist_ok=True)
    binary = replay.OUT / 'gz_visibility_capture_cleanup_fixed'
    dest = OUT / binary.name; shutil.copyfile(binary, dest); dest.chmod(0o755)
    paths.extend([binary, dest, ROOT/'tools/gz_visibility_capture_cleanup_fixed.cc'])
    frozen(path, dict(status='four_saved_frames_frozen', frames=frames, max_attempts=3,
        original_pixel_certification_requires_exact_RGB=True, labels_modified=False, training_started=False,
        inputs={str(p): file_sha256(p) for p in paths}))

async def run():
    p = read(OUT/'protocol.json'); verify(p)
    old = replay.OUT; oldlog = os.environ.get('GZ_LOG_PATH')
    lock = OUT/'replay.lock'; fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    replay.OUT = OUT; logs = OUT/'runtime-logs'; logs.mkdir(exist_ok=True); os.environ['GZ_LOG_PATH'] = str(logs)
    try:
        for f in p['frames']:
            for n in range(1,4):
                rp = OUT/'replay'/f['event_id']/f'attempt-{n:02}'/'receipt.json'
                if rp.exists():
                    r = read(rp); verify(r)
                elif rp.parent.exists(): continue
                else: r = await replay.attempt(f,n)
                if not r['process_cleanup_complete']: raise ValueError('Cleanup incomplete')
                if r['status'] != 'technical_failure': break
                log = rp.parent/'simulator.log'
                if log.exists() and 'Operation not permitted' in log.read_text(): break
    finally:
        replay.OUT = old
        if oldlog is None: os.environ.pop('GZ_LOG_PATH',None)
        else: os.environ['GZ_LOG_PATH'] = oldlog
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--freeze',action='store_true'); ap.add_argument('--replay',action='store_true'); a=ap.parse_args()
    if a.freeze: freeze()
    elif a.replay: asyncio.run(run())
    else: print('PREFLIGHT_ONLY_NO_REPLAY_NO_TRAINING')
