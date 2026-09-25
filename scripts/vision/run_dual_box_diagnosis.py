"""Same-pose full_2d / visible_2d diagnostic; existing sensor unchanged."""
import argparse,asyncio,copy,fcntl,os,shlex,subprocess
from pathlib import Path
import numpy as np
from scripts.vision.closed_body_material_pilot import OUT as SOURCE,ROOT,read,verify,frozen,file_sha256
from scripts.vision.instance_visibility_diagnosis import add_sensor,validate_world,tree_signature,ET,raw_box
import scripts.vision.run_visibility_cleanup_validation as replay
from src.sensors.gazebo_visual_transport import message_timestamp

OUT=SOURCE/'edge-box-trace-v1/dual-mode-v1'
ORIGINAL_ANALYZE=replay.analyze


def visible_sensor(tree):
    source=tree.find(".//sensor[@name='research_boxes']")
    if source is None or source.findtext('camera/box_type')!='full_2d':raise ValueError('Original full_2d required')
    sensor=copy.deepcopy(source);sensor.set('name','diagnostic_visible_boxes')
    sensor.find('topic').text='/diagnostic/visible_boxes';sensor.find('camera/box_type').text='visible_2d'
    return sensor


def dual_sensor(tree):
    result=add_sensor(tree)
    result.find(".//model[@name='canonical_camera']/link[@name='research_camera_link']").append(visible_sensor(tree))
    validate_dual(tree,result);return result


def validate_dual(original,derived):
    clean=copy.deepcopy(derived);found=clean.findall(".//sensor[@name='diagnostic_visible_boxes']")
    if len(found)!=1 or tree_signature(found[0])!=tree_signature(visible_sensor(original)):raise ValueError('Visible sensor differs from frozen alignment/type')
    clean.find(".//model[@name='canonical_camera']/link[@name='research_camera_link']").remove(found[0])
    validate_world(original,clean)


def parse_visible(raw,mapping):
    boxes={}
    for item in raw.get('annotatedBox',[]):
        label=str(int(item['label']))
        if label not in mapping or label in boxes:raise ValueError('Unknown or duplicate visible instance')
        box=raw_box(item)
        if not all(np.isfinite(v) and int(v)==v for v in box) or not (0<=box[0]<=box[2]<1920 and 0<=box[1]<=box[3]<1080):raise ValueError('Invalid visible pixel extrema')
        boxes[label]=dict(raw_inclusive=box,half_open=[box[0],box[1],box[2]+1,box[3]+1])
    return boxes


def analyze(folder,frame,fence):
    result=ORIGINAL_ANALYZE(folder,frame,fence)
    if result['status']!='original_pixel_evidence_certified':return result
    comparisons=[]
    for n in result['selected_capture_indices']:
        raw=read(folder/f'frame-{n}-visible-boxes.json');rgb=read(folder/f'frame-{n}-rgb.json')
        skew=abs(message_timestamp(raw)-message_timestamp(rgb))*1000
        if message_timestamp(raw)<=fence or skew>33.334+1e-6:raise ValueError('Visible sensor unsynchronized/stale')
        visible=parse_visible(raw,frame['instance_mapping']);full=read(folder/f'frame-{n}-boxes.json')
        mask=np.frombuffer((folder/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        ys,xs=np.where(mask[:,:,2]==128)
        comparisons.append(dict(capture_index=n,visible_skew_ms=skew,full_labels=[int(x['label']) for x in full.get('annotatedBox',[])],
            visible_boxes=visible,instance_128_pixels=len(xs),mask_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None))
    if any(c['visible_boxes']!=comparisons[0]['visible_boxes'] for c in comparisons):raise ValueError('Visible boxes unstable')
    result.update(status='dual_mode_diagnostic_complete',control_rgb_exact=True,comparisons=comparisons,
        original_training_labels_changed=False,reason='Original full_2d, RGB, pose and box-membership control passed; additional visible_2d output is diagnostic only.')
    return result


def freeze():
    path=OUT/'protocol.json'
    if path.exists():verify(read(path));return read(path)
    prior=SOURCE/'edge-box-trace-v1/completion.json';verify(read(prior));p=read(SOURCE/'protocol.json');verify(p)
    frame=next(f for f in p['frames'] if f['review_ids']==['T027']);OUT.mkdir(exist_ok=True)
    code=ROOT/'tools/gz_dual_box_capture.cc';binary=OUT/'gz_visibility_capture_cleanup_fixed'
    flags=subprocess.check_output(['pkg-config','--cflags','--libs','gz-transport13','gz-msgs10'],text=True)
    subprocess.run(['clang++',str(code),'-o',str(binary),*shlex.split(flags)],check=True,timeout=60)
    validate_dual(ET.parse(frame['source_world']),dual_sensor(ET.parse(frame['source_world'])))
    paths=[prior,SOURCE/'protocol.json',Path(__file__),Path(replay.__file__),ROOT/'scripts/vision/instance_visibility_diagnosis.py',code,binary]
    return frozen(path,dict(status='same_pose_dual_sensor_frozen',frames=[frame],max_attempts=3,
        policy='Keep original full_2d/RGB/depth and all world content; add only RGB-aligned instance sensor and copy of full_2d sensor with distinct name/topic and visible_2d box_type. No production mode change.',
        selection='Previously fixed T027; no pose changes or replacement images.',training_started=False,training_ready=False,
        inputs={str(p):file_sha256(p) for p in paths}))


async def run():
    p=freeze();dest=OUT/'render-completion.json'
    if dest.exists():verify(read(dest));print('VALID_DUAL_MODE_RUN_REUSED');return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze);oldlog=os.environ.get('GZ_LOG_PATH')
        replay.OUT=OUT;replay.add_sensor=dual_sensor;replay.validate_world=validate_dual;replay.analyze=analyze
        logs=OUT/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs);paths=[OUT/'protocol.json'];last=None
        try:
            for n in range(1,4):
                rp=OUT/'replay/T027'/f'attempt-{n:02}'/'receipt.json'
                if rp.exists():last=read(rp);verify(last)
                elif rp.parent.exists():continue
                else:last=await replay.attempt(p['frames'][0],n)
                paths.append(rp)
                if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
                if last['status']!='technical_failure':break
        finally:
            replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze=previous
            if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
            else:os.environ['GZ_LOG_PATH']=oldlog
        frozen(dest,dict(status=last['status'] if last else 'incomplete_attempts_exhausted',training_ready=False,training_started=False,
            inputs={str(p):file_sha256(p) for p in paths}))
        print(read(dest)['status'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:freeze();print('FROZEN_NO_RENDER_NO_TRAINING')
