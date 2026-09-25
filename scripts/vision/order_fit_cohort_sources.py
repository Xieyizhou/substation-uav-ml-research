"""Verify explicit cohort receipts, original pixels, full labels and viewed-dev poses."""
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.diagnose_small_scale_order_fit import OUT, TRAIN, prior
from scripts.vision.order_fit_legacy_pose_exclusion import distance
from scripts.vision.check_structure_fit_sources import equal_rgb, label_correspondence


def run():
    dest=OUT/'explicit-cohort-source-role-check.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',TRAIN/'design.json'];p,d=[prior.read(x) for x in paths]
    for r in (p,d):prior.verify(r)
    cache={};refs=[]
    def receipt(path):
        path=Path(path)
        if path not in cache:cache[path]=prior.read(path);paths.append(path)
        return cache[path]
    for kind in ['paired_review','negative_review']:
        rp=Path(d['evaluation'][kind]);r=prior.read(rp);prior.verify(r);paths.append(rp)
        for f in r['frames']:
            cp=Path(f['image_path']).parents[1]/'collection-receipt.json'
            vs=[v for v in receipt(cp)['views'] if v['view_id']==f['view_id']]
            if len(vs)!=1 or vs[0]['image_sha256']!=f['image_sha256']:raise ValueError('Development identity mismatch')
            refs.append(dict(view_id=f['view_id'],actual_pose=vs[0]['actual_pose'],receipt=str(cp)))
    rows=[]
    for m in p['members']:
        if not m.get('source_receipt'):continue
        row=dict(member_id=m['member_id'],registered_role=m['data_role'],lineage_id=m['lineage_id'],gaps=[])
        try:
            cp=Path(m['source_receipt']);r=receipt(cp);ip=Path(m.get('source_image_path') or m['source_image']);wp=Path(m['source_world'])
            vs=[v for v in r['views'] if Path(v['rgb_path']).resolve()==ip.resolve()]
            if len(vs)!=1:raise ValueError('Source frame missing or ambiguous')
            v=vs[0]
            if v['status']!='captured' or prior.file_sha256(ip)!=v['image_sha256']:raise ValueError('Captured RGB identity mismatch')
            for key in ('image','label'):
                if prior.file_sha256(m[key+'_path'])!=m[key+'_sha256']:raise ValueError('Member bytes changed')
                paths.append(Path(m[key+'_path']))
            equal_rgb(ip,m['image_path'])
            corresponding=label_correspondence(m['truth'],v['truth']['objects'])
            checks=r['collection_checks'];mapping=checks['instance_mapping']
            object_ids=[x['object_id'] for x in mapping.values()]
            if len(object_ids)!=len(set(object_ids)):raise ValueError('Instance mapping collision')
            if (checks['actual_annotation_mode'],checks['label_mode'],checks['hierarchy_mode'])!=('full_2d','visual-instance','top-level-equipment'):
                raise ValueError('Wrong receipt annotation semantics')
            world_sha=prior.file_sha256(wp)
            if r['world_sha256']!=world_sha or checks['world_sha256']!=world_sha:raise ValueError('World identity mismatch')
            modes=[x.text for x in ET.parse(wp).findall('.//box_type')]
            if not modes or set(modes)!={'full_2d'}:raise ValueError('Saved world annotation mode conflict')
            pose=v['actual_pose'];distance(pose,pose)
            hits=[]
            for ref in refs:
                metres,degrees=distance(pose,ref['actual_pose'])
                if metres<=.05 and degrees<=1:hits.append(dict(view_id=ref['view_id'],receipt=ref['receipt'],metres=metres,degrees=degrees))
            row.update(source_receipt=str(cp),source_image=str(ip),source_world=str(wp),world_sha256=world_sha,
                source_frame_id=v['view_id'],actual_pose={k:pose[k] for k in ['position','orientation']},
                matched_full_labels=len(corresponding),mapping_object_ids=object_ids,development_pose_matches=hits,
                status='source_verified_pose_role_review_required' if hits else 'source_pixels_full_labels_mapping_and_pose_screen_verified')
            paths.extend([ip,wp])
        except (KeyError,ValueError,FileNotFoundError,ET.ParseError) as exc:
            row['status']='source_check_blocked';row['gaps'].append(str(exc))
        rows.append(row)
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='explicit_cohort_sources_checked_not_dataset_admission',members=rows,
        counts=dict(Counter(x['status'] for x in rows)),
        limitations=['Source mapping and full label correspondence do not approve unlabelled structures.',
            'Pose coincidence flags a possible role conflict even across different layouts; absence does not prove independence.',
            'Shared assets/layouts retained; no protected labels read; no historical receipt re-certified.'],
        training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['counts'])
