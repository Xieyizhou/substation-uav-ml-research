"""Explicit same-frame diagnostic replay; no collection for training or admission."""
import argparse
import asyncio
import shutil
from pathlib import Path
from scripts.vision import run_instance_visibility_window as replay
from scripts.vision.record_pilot_full_label_audit import BASE
from src.vision.canonical.plan import read_record, write_record
from src.vision.canonical.gates import instance_mapping
from src.ml.artifacts import file_sha256

OUT=BASE/'pilot-full-label-replay-v1'
HELPER=Path('data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1/switchgear-condition-review-v1/instance-visibility-diagnosis-v1/gz_visibility_capture_window')

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():
        p=read_record(dest)
        for path,h in p['inputs'].items():
            if file_sha256(path)!=h:raise ValueError('Changed dependency')
        return p
    OUT.mkdir(parents=True,exist_ok=True)
    plan=read_record(BASE/'plan.json'); review=read_record(BASE/'pilot-full-label-audit-v1/full-label-review.json')
    rows=read_record(BASE/'pilot/collection-receipt.json')['views'];frames=[]
    paths=[BASE/'plan.json',BASE/'world.sdf',BASE/'obstacles.json',BASE/'pilot/collection-receipt.json',BASE/'pilot-full-label-audit-v1/full-label-review.json',Path(__file__),Path(replay.__file__),HELPER]
    mapping={str(k):v for k,v in instance_mapping(plan).items()}
    for i,v in enumerate(rows):
        selected=[b for b in review['boxes'] if b['view_id']==v['view_id'] and b['state']=='unresolved_content']
        if not selected:continue
        single=BASE/'pilot'/f'{v["view_id"]}.json'
        original=read_record(single)
        if original['raw_truth']!=v['raw_truth'] or original['actual_pose']!=v['actual_pose']:raise ValueError('Source disagreement')
        if file_sha256(v['rgb_path'])!=v['image_sha256']:raise ValueError('RGB changed')
        events=[]
        for j,b in enumerate(selected):
            label=next(int(k) for k,m in mapping.items() if m['object_id']==b['scene_device_id'])
            events.append(dict(review_id=f'frame-{i:02}-box-{j}',runtime_label=label,bbox_xyxy=b['bbox_xyxy']))
        frames.append(dict(member_id=v['view_id'],review_ids=[f'frame-{i:02}'],source_plan=str((BASE/'plan.json').resolve()),source_world=str((BASE/'world.sdf').resolve()),source_receipt=str(single.resolve()),source_image=v['rgb_path'],actual_pose=v['actual_pose'],instance_mapping=mapping,world_name=plan['world_name'],events=events))
        paths.extend([single,Path(v['rgb_path'])])
    shutil.copy2(HELPER,OUT/'gz_visibility_capture_window');paths.append(OUT/'gz_visibility_capture_window')
    return write_record(dest,dict(frames=frames,training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths}))

async def run():
    p=freeze();replay.OUT=OUT.resolve()
    # First affected frame validates replay before expanding to other frames.
    f=p['frames'][0]
    for n in range(1,4):
        folder=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}'
        if folder.exists():
            if not (folder/'receipt.json').exists():continue
            r=read_record(folder/'receipt.json')
            for path,h in r['inputs'].items():
                if file_sha256(path)!=h:raise ValueError('Stale replay')
        else:r=await replay.attempt(f,n)
        if r['status']!='technical_failure':return r
    return dict(status='technical_attempts_exhausted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');a=p.parse_args()
    if a.replay:print(asyncio.run(run())['status'])
    else:print(len(freeze()['frames']),'frozen; replay not started')
