"""Import explicit AI observations and export the fixed eight-source research cohort."""
import argparse
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from scripts.vision import supplement_closed_transformer as s
from scripts.vision.closed_exterior_review_observations import OBS,VARIANT_NOTES,HOLDS
from scripts.vision.record_closed_exterior_review import validate
from scripts.vision.same_source_material_quality_v2 import reference
from scripts.vision.export_material_candidate_batch import label_text
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash,REFERENCE_INDEXES
from src.ml.artifacts import object_sha256

prior=s.prior
OUT=s.base.OUT/'source-retention-control-v1'
PAIRS=('layout-A:closed:reactor:1','layout-A:closed:capacitor_bank:1','layout-A:closed:switchgear:1','layout-A:closed:transformer:1',
       'layout-B:closed:reactor:1','layout-B:closed:capacitor_bank:1','layout-B:closed:switchgear:1','layout-B:closed:transformer:supplement-1')
SUPPLEMENT_NOTES={
 'transformer_mid':'俯视下主体顶面、窄侧立面、基座边缘与三根顶部套管可辨，未见前景遮挡或图缘截断；侧面几何覆盖有限，不作为侧视来源。',
 'entry_switchgear':'斜俯视柜体顶面、侧面、面板边框和基座可辨；暖冷中性条件面板对比降低但边框仍可见，未见主体遮挡或截断。'}


def sources():
    for pair in PAIRS:
        new=pair.endswith('supplement-1');root=s.OUT if new else s.base.OUT
        for variant in s.base.old.PALETTES:
            key=variant if new else pair+':'+variant
            yield pair,variant,root/'evidence'/key/'evidence.json'


def record_review():
    OUT.mkdir(exist_ok=True);dest=OUT/'quality-review.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    rp=s.base.OUT/'original-replays-repaired-v1/completion.json';r=prior.read(rp);prior.verify(r)
    sp=s.OUT/'replays/completion.json';sr=prior.read(sp);prior.verify(sr)
    if sr['status']!='four_exact_replays_review_required':raise ValueError('Supplement replay incomplete')
    replays={x['pair_id']:Path(x['receipt']) for x in r['results']}
    supplement={x['variant']:Path(x['receipt']) for x in sr['results']}
    paths=[rp,sp,s.base.OUT/'visual-review.json',Path(__file__),Path(__file__).with_name('closed_exterior_review_observations.py')]
    es=[];decisions=[];members=[]
    for pair,variant,ep in sources():
        if pair in HOLDS:raise ValueError('Held group selected')
        e=prior.read(ep);prior.verify(e);es.append(e);paths.append(ep)
        new=pair.endswith('supplement-1');replay_path=supplement[variant] if new else replays[pair]
        result=prior.read(replay_path);prior.verify(result);paths.append(replay_path)
        if result['status']!='original_pixel_evidence_certified' or any(x['missing_targets'] for x in result['full_mask_coverage']):raise ValueError('Replay gap')
        notes=SUPPLEMENT_NOTES if new else OBS[pair]
        if set(notes)!={x['object_id'] for x in e['events']}:raise ValueError('Full review inventory changed')
        original=prior.read(ep.parent.parent/('original' if new else pair+':original')/'evidence.json')
        if {x['object_id']:x['truth']['bbox_xyxy'] for x in original['events']}!={x['object_id']:x['truth']['bbox_xyxy'] for x in e['events']}:raise ValueError('Paired full boxes changed')
        plan=Path(e['unit']['plan_path']);orig_plan=plan.parent.parent/'original/plan.json'
        p=prior.read(plan);op=prior.read(orig_plan)
        if p['objects']!=op['objects'] or p['calibration_views']!=op['calibration_views']:raise ValueError('Geometry/pose variation')
        names={o['name'] for o in p['objects'] if o['category'] in s.base.old.COUNTS}
        s.base.old.check_only_materials(ET.parse(orig_plan.parent/'world.sdf').getroot(),ET.parse(plan.parent/'world.sdf').getroot(),names)
        paths.extend([plan,orig_plan,plan.parent/'world.sdf'])
        for x in e['events']:
            decisions.append(dict(event_id=x['event_id'],object_id=x['object_id'],status='approved_for_bounded_research_cohort',
                reason=notes[x['object_id']]+' '+VARIANT_NOTES[variant]+' 实例全图覆盖经绑定重放核对；旧组材质帧仅使用原始重放辅助证据，不传播原帧像素认证。',
                review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
                crop_sha256=x['crop_sha256'],truth_identity=object_sha256(x['truth']),evidence_identity=e['identity'],
                pixel_visibility_certified=new or variant=='original',training_admitted=False,promotable=False))
        members.append(dict(member_id='new-'+object_sha256({'pair':pair,'variant':variant})[:24],pair_id=pair,variant=variant,
            image_path=e['image_path'],image_sha256=e['image_sha256'],full_truth=e['full_truth'],evidence_path=str(ep),
            actual_pose=e['actual_pose'],source_world=str(plan.parent/'world.sdf'),source_receipt=e['source_receipt'],
            planned_category=e['unit']['category'],planned_object_id=e['unit']['object_id'],instance_mapping=e['instance_mapping'],
            layout=pair.split(':')[0],lineage_id=pair,training_admitted=False,promotable=False))
    validate(es,decisions)
    return prior.frozen(dest,dict(status='eight_sources_explicitly_reviewed_research_only',members=members,decisions=decisions,
        held_pairs=sorted(HOLDS),selection_rule='First registered non-held planned source per class per layout; B transformer uses the separately authorized supplement. Eight sources fit the bounded 30 slots while retaining original exposure. Other candidates not automatically admitted.',
        independent_pose_groups=8,layouts=2,asset_independence=False,training_ready=False,training_started=False,training_admitted=False,promotable=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))


def export():
    review=prior.read(OUT/'quality-review.json');prior.verify(review)
    evidence=[prior.read(ep) for _,_,ep in sources()]
    for e in evidence:prior.verify(e)
    validate(evidence,review['decisions'])
    if any(d['status']!='approved_for_bounded_research_cohort' for d in review['decisions']):raise ValueError('Unapproved member')
    dest=OUT/'export/manifest.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    p,_,_=reference.contract('T-7');refs=list(p['pool_rows']);paths=[OUT/'quality-review.json',reference.OUT/'protocol.json',Path(__file__)]
    for key in ('paired_review','negative_review'):
        path=Path(p['evaluation'][key]);q=prior.read(path);refs+=q['frames'];paths.append(path)
    hashes=set();pixels=set()
    for ref in refs:
        path=Path(ref['image_path']);sha=prior.file_sha256(path)
        if sha!=ref['image_sha256']:raise ValueError('Stale reference')
        hashes.add(sha);pixels.add(pixel_hash(path));paths.append(path)
    for path in REFERENCE_INDEXES:
        paths.append(path)
        for line in path.read_text().splitlines():
            x=json.loads(line);hashes.add(x.get('image_sha256'));pixels.add(x.get('pixel_sha256'))
    rows=[];seen={};gaps=[]
    for m in review['members']:
        ip=Path(m['image_path']);px=pixel_hash(ip)
        if prior.file_sha256(ip)!=m['image_sha256']:raise ValueError('Candidate changed')
        if m['image_sha256'] in hashes or px in pixels or px in seen:gaps.append(dict(member=m['member_id'],reason='exact_reference_or_candidate_overlap'))
        seen[px]=m['member_id'];image=Image.open(ip).convert('RGB')
        op=OUT/'export/images'/(m['member_id']+'.png');lp=OUT/'export/labels'/(m['member_id']+'.txt')
        op.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
        text=label_text(m['full_truth'],*image.size)
        if op.exists():
            if pixel_hash(op)!=px:raise ValueError('Export pixels differ')
        else:image.save(op)
        if lp.exists():
            if lp.read_text()!=text:raise ValueError('Export labels differ')
        else:lp.write_text(text)
        if pixel_hash(op)!=px:raise ValueError('Lossless conversion failed')
        rows.append(dict(m,source_image=m['image_path'],image_path=str(op),image_sha256=prior.file_sha256(op),pixel_sha256=px,
            label_path=str(lp),label_sha256=prior.file_sha256(lp),class_instances=dict(Counter(o['class_name'] for o in m['full_truth']['objects'])),
            subset='bridge_positive',data_role='bounded_development_training_candidate',lineage_resolution='layout_instance_pose_derived_group',
            asset_independence=False,source_independent=False))
        paths.extend([ip,op,lp])
    return prior.frozen(dest,dict(status='reference_overlap_blocked' if gaps else 'lossless_export_exclusion_passed',members=rows,gaps=gaps,
        reference_images_checked=len(refs),protected_labels_read=False,shared_assets=True,training_admitted=False,promotable=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--record-review',action='store_true');a=ap.parse_args()
    if a.record_review:print(record_review()['status'])
    else:print(export()['status'])
