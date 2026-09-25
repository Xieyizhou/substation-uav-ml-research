"""Evidence-only supervision proposal: never export labels or train."""
import argparse
import asyncio
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
from scripts.vision.structure_fit import OUT as SOURCE,ROOT,read,verify,verify_tree,frozen,file_sha256,unique,truth_for
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence
from scripts.vision.instance_visibility_diagnosis import raw_box
from scripts.vision import run_visibility_cleanup_validation as replay
from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world
from src.vision.collection.gazebo_truth import _annotation
from src.ml.artifacts import object_sha256

OUT=SOURCE.parent/'supervision-risk-revision-v1'
IDS=('T30','T08','T29','T33','T04','T26','T32')

def freeze():
    if (OUT/'protocol.json').exists():verify_tree(OUT/'protocol.json');return read(OUT/'protocol.json')
    verify_tree(SOURCE/'completion.json');OUT.mkdir(exist_ok=True)
    p=read(SOURCE/'protocol.json');s=read(SOURCE/'member-source-trace.json');src=unique(s['rows'],lambda r:r['member_id'])
    paths=[SOURCE/'completion.json',SOURCE/'protocol.json',SOURCE/'member-source-trace.json',SOURCE/'review.json',ROOT/'docs/supervision-risk-revision-plan-v1.md',Path(__file__),
        ROOT/'src/vision/collection/gazebo_truth.py',ROOT/'src/vision/canonical/gates.py',ROOT/'scripts/vision/run_visibility_cleanup_validation.py',ROOT/'scripts/vision/instance_visibility_diagnosis.py',ROOT/'tools/gz_visibility_capture_cleanup_fixed.cc']
    inputs={str(x):file_sha256(x) for x in paths};frames=[]
    binary=replay.OUT/'gz_visibility_capture_cleanup_fixed';shutil.copyfile(binary,OUT/binary.name);(OUT/binary.name).chmod(0o755)
    inputs[str(binary)]=file_sha256(binary);inputs[str(OUT/binary.name)]=file_sha256(OUT/binary.name)
    prior=read(replay.OUT/'protocol.json');inputs[str(replay.OUT/'protocol.json')]=file_sha256(replay.OUT/'protocol.json')
    for eid in IDS:
        e=next(e for e in p['reactors'] if e['event_id']==eid);m=e['member'];sr=src[m['member_id']];ip=Path(sr['source_image']);rp=ip.parents[1]/(ip.parent.name+'.json');pp=ip.parents[2]/'plan/plan.json'
        rec=read(rp);plan=read(pp);wp=pp.parent/'world.sdf';mapping=instance_mapping(plan)
        for path in (rp,pp,wp,pp.parent/'sensor_source.sdf',pp.parent/'obstacles.json',ip,Path(m['image_path']),Path(m['label_path']),SOURCE/f'{eid}.png'):
            inputs[str(path)]=file_sha256(path)
        if file_sha256(wp)!=plan['files']['world.sdf']:raise ValueError('World changed')
        equal_rgb(ip,m['image_path']);label_correspondence(truth_for(m),rec['truth']['objects'])
        matches=[b for b in rec['raw_truth']['annotatedBox'] if max(abs(a-c) for a,c in zip(raw_box(b),e['truth']['bbox_xyxy']))<1e-4]
        if len(matches)!=1:raise ValueError('Ambiguous raw target')
        b=matches[0];label=int(b['label'])
        if label not in mapping or mapping[label]['category']!='reactor':raise ValueError('Unresolved target instance')
        with Image.open(ip) as im:w,h=im.size
        recomputed=_annotation(b,index=e['truth']['label_line_index'],width=w,height=h,message_id='semantic-recomputation').to_record()
        original=next(t for t in rec['truth']['objects'] if t['class_name']=='reactor')
        if recomputed['truncation_status']!=original['truncation_status']:raise ValueError('Truncation conversion not reproduced')
        f=dict(event_id=eid,member_id=m['member_id'],member=m,review_ids=[eid],truth=e['truth'],source_image=str(ip),source_plan=str(pp),source_world=str(wp),source_receipt=str(rp),
            world_name=plan['world_name'],actual_pose=rec['actual_pose'],instance_mapping={str(k):v for k,v in mapping.items()},
            mapping_basis='posthoc_saved_plan_not_original_gate_certificate' if sr['gaps'] else 'original_receipt_and_saved_plan',
            annotation_mode=annotation_mode_from_world(wp),historical_truncation_status=original['truncation_status'],raw_target=b,
            converter_recomputed_truncation=recomputed['truncation_status'],truncation_semantics='whether converter clamps incoming box; not physical object truncation certification',
            semantic_limit='Current converter exactly reproduces saved values; original collector revision does not pin a source-code hash.',
            events=[dict(review_id=eid,runtime_label=label,object_id=mapping[label]['object_id'],bbox_xyxy=e['truth']['bbox_xyxy'])],source_metadata_gaps=sr['gaps'])
        # Build fresh evidence with all full-image labels, no predictions or model results.
        im=Image.open(ip).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full)
        for t in truth_for(m):draw.rectangle(t['bbox_xyxy'],outline='lime' if t['annotation_id']==e['truth']['annotation_id'] else 'orange',width=3)
        page=Image.new('RGB',(1200,600),'white');full.thumbnail((800,540));page.paste(full,(0,50))
        box=e['truth']['bbox_xyxy'];crop=im.crop(tuple(box));crop.thumbnail((390,540));page.paste(crop,(805,50));ImageDraw.Draw(page).text((5,5),eid+' full truth; no predictions',fill='black')
        ep=OUT/f'{eid}.png';page.save(ep);inputs[str(ep)]=file_sha256(ep);f.update(evidence_path=str(ep),evidence_sha256=file_sha256(ep))
        existing=[r for r in prior['frames'] if r['source_image']==str(ip)]
        f['existing_replay_candidates']=[str(path) for x in existing for path in (replay.OUT/'replay'/x['review_ids'][0]).glob('attempt-*/receipt.json')]
        frames.append(f)
    return frozen(OUT/'protocol.json',dict(status='frozen_evidence_only',frames=frames,inputs=inputs,pilot=['T30','T08'],
        scope='7 unique labels; original files immutable; no training/data export',model_outputs_hidden_in_new_review_evidence=True,
        prior_model_results_already_known_not_a_blind_study=True,policy=dict(max_attempts=3,position_m=.05,attitude_deg=1,skew_ms=33.334,stable_frames=3,box_delta_px=1,rgb_exact_required=True)))

async def run_replays():
    p=read(OUT/'protocol.json');verify(p)
    # Reuse the process/camera/sync implementation with only an independent output root.
    original_out=replay.OUT;replay.OUT=OUT
    async def unit(f):
        for n in range(1,4):
            rp=OUT/'replay'/f['event_id']/f'attempt-{n:02}'/'receipt.json'
            if rp.exists():verify_tree(rp);r=read(rp)
            elif rp.parent.exists():continue
            else:r=await replay.attempt(f,n)
            if r['status']!='technical_failure':return r
        return dict(status='technical_attempts_exhausted',reason='Three attempts consumed')
    try:
        pilot=[await unit(f) for f in p['frames'][:2]]
        if all(r['status']=='original_pixel_evidence_certified' for r in pilot):
            for f in p['frames'][2:]:await unit(f)
        else:print('PILOT_NOT_CERTIFIED_NO_EXPANSION',flush=True)
    finally:replay.OUT=original_out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.freeze:freeze();print('FROZEN_SEVEN')
    elif a.replay:asyncio.run(run_replays())
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
