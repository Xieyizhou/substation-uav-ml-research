"""Lossless explicitly reviewed counterpart export with metadata-only exclusions."""
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.physical_low_light_capture import OUT,COHORT,freeze,prior
from scripts.vision.verify_physical_low_light import verify_capture,evidence
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash,REFERENCE_INDEXES
from scripts.vision.export_material_candidate_batch import label_text
from scripts.vision.closed_budget_design import freeze as base_design

def validate_review(p,r):
    prior.verify(r);want={};frames={}
    for u in p['units']:
        e=evidence(u);frames[u['unit_id']]=e
        for x in e['events']:want[x['event_id']]=(e,x)
    ds=r['decisions']
    if len(ds)!=len(want) or {d['event_id'] for d in ds}!=set(want) or set(r['full_frames_viewed'])!=set(frames):raise ValueError('Missing/duplicate full review')
    for d in ds:
        e,x=want[d['event_id']]
        if d['evidence_identity']!=e['identity'] or d['crop_sha256']!=x['crop_sha256']:raise ValueError('Stale explicit review')
        if d['status']!='content_sufficient_for_bounded_research' or not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Unapproved/unknown content')
        if d['training_admitted'] or d['promotable']:raise ValueError('Invalid promotion')
    return frames

def export():
    p=freeze();rp=OUT/'quality-review.json';r=prior.read(rp);frames=validate_review(p,r);dest=OUT/'export/manifest.json'
    if dest.exists():m=prior.read(dest);prior.verify(m);return m
    old=base_design();refs=list(old['pool_rows']);deps=[OUT/'capture-protocol.json',rp,COHORT/'quality-review.json',Path(__file__).resolve()]
    for name in ('paired_review','negative_review'):
        path=Path(old['evaluation'][name]);review=prior.read(path);prior.verify(review);refs+=review['frames'];deps.append(path)
    hashes=set();pixels=set()
    for row in refs:
        path=Path(row['image_path']);digest=prior.file_sha256(path)
        if digest!=row['image_sha256']:raise ValueError('Reference drift')
        hashes.add(digest);pixels.add(pixel_hash(path));deps.append(path)
    for index in REFERENCE_INDEXES:
        deps.append(index)
        for line in index.read_text().splitlines():
            m=json.loads(line);hashes.add(m.get('image_sha256'));pixels.add(m.get('pixel_sha256'))
    members=[];seen=set()
    for u in p['units']:
        uid=u['unit_id'];e=frames[uid];cp=OUT/'replays'/uid/'completion.json';replay=prior.read(cp);prior.verify(replay)
        if replay['status']!='low_light_capture_exact_replay_verified':raise ValueError('Replay not certified')
        row,mapping,source=verify_capture(u);ip=Path(row['rgb_path']);px=pixel_hash(ip)
        if row['image_sha256'] in hashes or px in pixels or px in seen:raise ValueError('Exact reference/candidate overlap')
        seen.add(px);member='low-'+u['source_member']['member_id'];image=OUT/'export/images'/f'{member}.png';label=OUT/'export/labels'/f'{member}.txt'
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        text=label_text(row['truth'],1920,1080)
        if text!=Path(u['source_member']['label_path']).read_text():raise ValueError('Full paired labels changed')
        if image.exists():
            if pixel_hash(image)!=px:raise ValueError('Export pixel mismatch')
        else:Image.open(ip).convert('RGB').save(image)
        if label.exists():
            if label.read_text()!=text:raise ValueError('Export labels changed')
        else:label.write_text(text)
        if pixel_hash(image)!=px:raise ValueError('Lossless export failed')
        members.append(dict(u['source_member'],member_id=member,source_member_id=u['source_member']['member_id'],illumination='physical_low_neutral',image_path=str(image),image_sha256=prior.file_sha256(image),pixel_sha256=px,
            label_path=str(label),label_sha256=prior.file_sha256(label),source_image=str(ip),source_world=str(Path(u['plan_path']).parent/'world.sdf'),source_receipt=str(source),evidence_path=str(OUT/'evidence'/uid/'evidence.json'),
            actual_pose=row['actual_pose'],instance_mapping=mapping,full_truth=row['truth'],class_instances=dict(Counter(t['class_name'] for t in row['truth']['objects'])),
            data_role='bounded_development_training_candidate',source_independent=False,asset_independence=False,training_admitted=False,promotable=False))
        deps.extend([image,label,ip,source,cp,OUT/'evidence'/uid/'evidence.json'])
    return prior.frozen(dest,dict(status='32_reviewed_lossless_counterparts_isolation_checked',members=members,reference_images_checked=len(refs),protected_labels_read=False,
        independent_pose_groups=8,new_independent_pose_groups=0,layouts=2,shared_assets=True,derivation_policy='Each lower-light counterpart stays in the same training role and lineage as its original. No development or protected image used for training.',inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':print(export()['status'])
