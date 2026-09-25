"""Same-frame full-target coverage for four existing physical-lighting renders."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior
from scripts.vision.audit_material_mask_coverage import coverage
from scripts.vision.test_body_material_applicability import model_mapping
from scripts.vision.check_structure_fit_sources import equal_rgb
from scripts.vision.order_fit_legacy_pose_exclusion import distance
from scripts.vision.run_physical_lighting_capture_v2 import base


def run():
    dest=OUT/'lighting-full-mask-coverage.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',OUT/'source-resolution.json'];p,s=[prior.read(x) for x in paths]
    for r in (p,s):prior.verify(r)
    pool={m['member_id']:m for m in p['members']};rows=[]
    for source in s['resolved_sources']:
        m=pool[source['member_id']];ip=Path(source['actual_render_image']);rp=Path(source['receipt']);r=prior.read(rp);prior.verify(r)
        equal_rgb(ip,m['image_path']);wp=rp.parent/'world.sdf'
        if r['inputs'].get(str(wp))!=prior.file_sha256(wp):raise ValueError('World not bound')
        mapping,_=model_mapping(ET.parse(wp));mapping={str(k):dict(object_id=v) for k,v in mapping.items()}
        if len(r['records'])!=3 or not r['process_cleanup_complete']:raise ValueError('Incomplete stable replay')
        if r['inputs'].get(str(ip))!=prior.file_sha256(ip):raise ValueError('Rendered RGB not bound')
        checks=[]
        for record in r['records']:
            # Frozen protocol pairs each stream to RGB; total earliest/latest span
            # is descriptive and is not an additional admission threshold.
            metas=[rp.parent/f"frame-{record['capture_index']}-{kind}.json" for kind in ('rgb','depth','mask','pose','boxes','visible-boxes')]
            for path in metas:
                if r['inputs'].get(str(path))!=prior.file_sha256(path):raise ValueError('Unbound stream timestamp')
            stamps=[base.message_timestamp(prior.read(path)) for path in metas]
            skew=max(abs(t-stamps[0]) for t in stamps)*1000
            if abs(skew-record['skew_ms'])>1e-6 or skew>33.334001 or max(record['historical_deltas'].values())>1:raise ValueError('Alignment failed')
            paths.extend(metas)
            metres,degrees=distance(record['actual_pose'],r['records'][0]['actual_pose'])
            if metres>.05 or degrees>1:raise ValueError('Unstable pose')
            boxes=record['full_boxes'];observed=sorted(tuple(round(v,4) for v in b) for b in boxes.values())
            expected=sorted(tuple(round(v,4) for v in t['bbox_xyxy']) for t in m['truth'])
            if len(observed)!=len(expected) or any(max(abs(a-b) for a,b in zip(x,y))>1e-3 for x,y in zip(observed,expected)):raise ValueError('Incomplete current full labels')
            mp=rp.parent/f"frame-{record['capture_index']}-mask.bin"
            if r['inputs'].get(str(mp))!=prior.file_sha256(mp):raise ValueError('Unbound mask')
            mask=np.fromfile(mp,dtype='u1').reshape(1080,1920,3);result=coverage(mask,list(map(int,boxes)),mapping)
            if result['missing_targets']:raise ValueError('Unboxed target pixels')
            checks.append(dict(capture_index=record['capture_index'],rgb_referenced_skew_ms=skew,all_stream_span_ms=(max(stamps)-min(stamps))*1000,visible_pixels_by_label={str(k):v for k,v in result['visible_pixels_by_label'].items()},unboxed_target_count=0));paths.append(mp)
        rows.append(dict(member_id=m['member_id'],status='same_frame_lighting_full_target_coverage_verified',checks=checks,
            image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],review_decisions=source['review_decisions']))
        paths.extend([ip,rp,wp,Path(m['image_path']),Path(m['label_path'])])
    paths.extend([Path(__file__).resolve(),Path(__file__).with_name('audit_material_mask_coverage.py'),Path(base.__file__).resolve(),Path(__file__).with_name('run_physical_lighting_capture_v2.py')])
    return prior.frozen(dest,dict(status='four_lighting_frames_coverage_verified',members=rows,
        scope='Current rendered frames only; no certification of original-parent pixels. Existing explicit AI content limits retained.',
        training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
