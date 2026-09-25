"""Lossless full-label export, source gates and exact role exclusion for reviewed captures."""
from collections import Counter
from pathlib import Path
from PIL import Image
from src.ml.artifacts import file_sha256,object_sha256
from src.vision.canonical.plan import write_record,pose_close
from src.vision.canonical.gates import validate_preflight,validate_view_pose,target_checks
from src.vision.training.hard_example_curator import dhash64
from scripts.vision.prepare_clear_context_increment import OUT,checked
from scripts.vision.review_clear_context_increment import run as reviewed,validate
from scripts.vision.reviewed_negative_order_control import OUT as CONTROL
from scripts.vision.export_material_candidate_batch import label_text
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash,load_fingerprints
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs


def export():
    dest=OUT/'dataset/manifest.json'
    if dest.exists():return checked(dest)
    review=reviewed();e=checked(OUT/'evidence/manifest-v2.json');validate(e,review)
    pp=OUT/'plan/plan.json';plan=checked(pp);cp=OUT/'capture-attempt-003/collection-receipt.json';capture=checked(cp)
    gate,config,mapping=validate_preflight(plan,pp.parent,plan['calibration_views'])
    if capture['collection_checks']!=gate or capture['status']!='complete_pending_review':raise ValueError('Capture checks changed')
    views={v['view_id']:v for v in plan['calibration_views']};frames={v['view_id']:v for v in capture['views']}
    control=checked(CONTROL/'protocol.json');paired,negative,dependencies=_load_inputs()
    refs=[dict(member_id=r['member_id'],role='current_training_pool',image_path=r['image_path'],image_sha256=r['image_sha256']) for r in control['pool_rows']]
    refs += [dict(member_id=r['view_id'],role='fixed_paired_development',image_path=r['image_path'],image_sha256=r['image_sha256']) for r,_ in paired]
    refs += [dict(member_id=r['view_id'],role='fixed_negative_development',image_path=r['image_path'],image_sha256=r['image_sha256']) for r in negative]
    files,pixels,protected_inputs=load_fingerprints();dependencies.update(protected_inputs)
    reference_records=[]
    for r in refs:
        if file_sha256(r['image_path'])!=r['image_sha256']:raise ValueError('Reference image drift')
        ph=pixel_hash(r['image_path']);dh=dhash64(r['image_path']);files.add(r['image_sha256']);pixels.add(ph)
        reference_records.append(dict(**r,pixel_sha256=ph,dhash64=dh));dependencies[r['image_path']]=r['image_sha256']
    root=dest.parent;(root/'images').mkdir(parents=True,exist_ok=True);(root/'labels').mkdir(exist_ok=True)
    members=[];near=[];seen=set()
    for d in review['frames']:
        frame=frames[d['view_id']];v=views[d['view_id']];rid=d['review_id']
        if object_sha256(frame['truth'])!=d['truth_sha256']:raise ValueError('Full truth differs from reviewed evidence')
        if not pose_close(frame['actual_pose'],v):raise ValueError('Actual pose mismatch')
        validate_view_pose(v,config,actual_carrier=frame['actual_pose']['position'])
        tc=target_checks(v,frame['raw_truth'],mapping)
        if tc!=frame['target_checks'] or tc['planned_instance_present'] is not True:raise ValueError('Instance check mismatch')
        times=[frame[k] for k in ('rgb_timestamp','depth_timestamp','truth_timestamp')]+[frame['actual_pose']['timestamp']]
        if max(abs(times[0]-x) for x in times[1:])>.033334+1e-9:raise ValueError('Synchronization mismatch')
        source=Path(frame['rgb_path']);ph=pixel_hash(source);sha=file_sha256(source)
        if sha!=d['image_sha256'] or sha in files or ph in pixels or ph in seen:raise ValueError('Duplicate or excluded reference pixels')
        seen.add(ph);image=Image.open(source).convert('RGB');text=label_text(frame['truth'],*image.size)
        ip=root/'images'/f'{rid}.png';lp=root/'labels'/f'{rid}.txt'
        if ip.exists():
            if pixel_hash(ip)!=ph:raise ValueError('Existing image changed')
        else:image.save(ip)
        if lp.exists():
            if lp.read_text()!=text:raise ValueError('Existing labels changed')
        else:lp.write_text(text)
        if pixel_hash(ip)!=ph:raise ValueError('Lossless export failed')
        dh=dhash64(source)
        nearest=min(reference_records,key=lambda r:(int(dh,16)^int(r['dhash64'],16)).bit_count())
        distance=(int(dh,16)^int(nearest['dhash64'],16)).bit_count()
        near.append(dict(review_id=rid,distance=distance,reference=nearest,review_status='needs_pair_review' if distance<=2 else 'no_close_dhash_trigger'))
        members.append(dict(member_id='clear-context:'+v['view_id'],review_id=rid,subset='bridge_positive',variant='original',
            image_path=str(ip.resolve()),label_path=str(lp.resolve()),image_sha256=file_sha256(ip),label_sha256=file_sha256(lp),pixel_sha256=ph,
            source_image_path=str(source),source_image_sha256=sha,source_receipt=str(cp.resolve()),source_plan=str(pp.resolve()),
            lineage_id='extreme:pose:'+v['view_id'],pair_id=v['view_id'],planned_object_id=v['object_id'],planned_category=v['category'],
            actual_pose=frame['actual_pose'],class_instances=dict(Counter(o['class_name'] for o in frame['truth']['objects'])),truth=frame['truth']['objects'],
            layout='existing_extreme_development_layout',asset_family='shared_canonical_primitive_equipment',independent_scene_claim=False,
            full_frame_evidence=str((OUT/'review.json').resolve()),review_identity=review['identity'],training_admitted=False,promotable=False))
        for path in (source,ip,lp):dependencies[str(path.resolve())]=file_sha256(path)
    for p in (pp,cp,OUT/'review.json',OUT/'evidence/manifest-v2.json',OUT/'runtime-snapshot.json',CONTROL/'protocol.json',Path(__file__)):
        dependencies[str(p.resolve())]=file_sha256(p)
    return write_record(dest,dict(status='reviewed_lossless_export_pending_near_duplicate_review' if any(r['distance']<=2 for r in near) else 'reviewed_lossless_export_and_reference_exclusion_passed',
        members=members,reference_count=len(refs),nearest_references=near,protected_labels_read=False,
        new_camera_pose_count=12,new_layout_count=0,new_asset_count=0,training_admitted=False,promotable=False,inputs=dependencies))


if __name__=='__main__':
    r=export();print(r['status'],[(x['review_id'],x['distance'],x['reference']['role']) for x in r['nearest_references']])
