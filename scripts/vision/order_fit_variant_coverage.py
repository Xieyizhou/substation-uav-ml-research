"""Recheck existing same-frame masks for unboxed targets; reuse explicit content review."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior
from scripts.vision.audit_material_mask_coverage import coverage
from scripts.vision.test_body_material_applicability import model_mapping
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence
from scripts.vision.order_fit_legacy_pose_exclusion import distance


def run():
    dest=OUT/'appearance-full-mask-coverage.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',OUT/'early-positive-review-links.json',OUT/'closeout-remaining-quality.json',
           OUT/'gray-full-label-correspondence.json',OUT/'variant-source-role-check.json']
    p,links,*rest=[prior.read(path) for path in paths]
    for r in (p,links,*rest):prior.verify(r)
    pool={m['member_id']:m for m in p['members']};originals={};rows=[]
    for link in links['members']:
        if not re.fullmatch(r'[GS]\d\d-original',link['member_id']):continue
        rp=Path(link['decisions'][0]['review_path']);r=prior.read(rp);prior.verify(r);paths.append(rp)
        sid=link['member_id'].split('-')[0];sources=[s for s in r['sources'] if s['probe_id']==sid]
        if len(sources)!=1:raise ValueError('Ambiguous original source')
        s=sources[0]
        if not s['exact_replay'] or s['unboxed_target_count']!=0 or not s['mask_inside_full_boxes']:raise ValueError('Original coverage not established')
        m=pool[link['member_id']];equal_rgb(s['image_path'],m['image_path']);label_correspondence(m['truth'],s['full_truth']['objects'])
        originals[sid]=s;paths.extend([Path(s['image_path']),Path(m['image_path']),Path(m['label_path'])])
        rows.append(dict(member_id=m['member_id'],status='existing_exact_source_full_coverage_revalidated',review_source=str(rp),unboxed_target_count=0))
    for mid,m in pool.items():
        match=re.fullmatch(r'([GS]\d\d)-(warm|cool|gray_target_body|gray_all_body)',mid)
        if not match:continue
        source=originals[match[1]];ip=Path(m['source_image']);root=ip.parent;rp=root.parent/'receipt.json'
        r=prior.read(rp);prior.verify(r);paths.append(rp)
        if r['stable_frames']!=3 or not r['process_cleanup_complete']:raise ValueError('Replay incomplete')
        if r['inputs'].get(str(ip))!=prior.file_sha256(ip):raise ValueError('RGB absent from replay identity')
        equal_rgb(ip,m['image_path']);label_correspondence(m['truth'],source['full_truth']['objects'])
        wp=root.parent/'world.sdf'
        if r['inputs'].get(str(wp))!=prior.file_sha256(wp):raise ValueError('Actual world binding missing')
        mapping,_=model_mapping(ET.parse(wp));expected={str(k):v['object_id'] for k,v in source['instance_mapping'].items()}
        if mapping!=expected:raise ValueError('Runtime instance mapping changed')
        objects=source['full_truth']['objects'];labels=[];boxes={}
        for obj in objects:
            label=int(re.search(r'instance-(\d+)-',obj['annotation_id'])[1]);labels.append(label);boxes[label]=obj['bbox_xyxy']
        checks=[]
        for n in range(1,4):
            rec=[v for v in r['records'] if v['frame_index']==n]
            if len(rec)!=1:raise ValueError('Missing stable frame record')
            rec=rec[0];metres,degrees=distance(rec['actual_pose'],source['actual_pose'])
            if metres>.05 or degrees>1 or rec['skew_ms']>33.334 or rec['maximum_box_delta_px']>1:raise ValueError('Pose/sync/box drift')
            mp=root/f'frame-{n}-mask.bin'
            if r['inputs'].get(str(mp))!=prior.file_sha256(mp):raise ValueError('Mask absent from receipt identity')
            mask=np.fromfile(mp,dtype='u1').reshape(1080,1920,3);check=coverage(mask,labels,source['instance_mapping'])
            if check['missing_targets']:raise ValueError('Unboxed visible target: '+mid)
            for label in check['visible_pixels_by_label']:
                yy,xx=np.where(mask[:,:,2]==label);a,b,c,d=boxes[label]
                if xx.min()<a-1 or yy.min()<b-1 or xx.max()+1>c+1 or yy.max()+1>d+1:raise ValueError('Visible mask outside full label')
            checks.append(dict(frame=n,visible_pixels_by_label={str(k):v for k,v in check['visible_pixels_by_label'].items()},unboxed_target_count=0))
            paths.append(mp)
        rows.append(dict(member_id=mid,status='same_frame_full_mask_coverage_verified_content_review_separate',checks=checks,
            replay_receipt=str(rp),image_sha256=m['image_sha256'],label_sha256=m['label_sha256']))
        paths.extend([ip,wp,Path(m['image_path']),Path(m['label_path'])])
    if len(rows)!=60 or len({m['member_id'] for m in rows})!=60:raise ValueError('Incomplete 60-frame coverage')
    paths.extend([Path(__file__).resolve(),Path(__file__).with_name('audit_material_mask_coverage.py'),Path(__file__).with_name('test_body_material_applicability.py')])
    return prior.frozen(dest,dict(status='60_appearance_frames_full_target_coverage_verified',members=rows,
        scope='12 original exact-source evidence checks plus 48 variants x 3 same-frame masks. No parent RGB pixel certification is transferred to variants. Explicit per-label quality decisions remain separate.',
        training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
