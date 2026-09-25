"""Resolve final source frames and actual poses without inheriting parent pixels."""
from collections import Counter
from pathlib import Path
import re
from scripts.vision.diagnose_small_scale_order_fit import OUT,TRAIN,prior
from scripts.vision.order_fit_legacy_pose_exclusion import distance
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence


def run():
    dest=OUT/'variant-source-role-check.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',OUT/'source-resolution.json',TRAIN/'design.json',
           OUT/'gray-full-label-correspondence.json',OUT/'early-positive-review-links.json',OUT/'c01-source-correspondence.json']
    records=[prior.read(p) for p in paths]
    for r in records:prior.verify(r)
    p,resolved,design=records[:3];light={m['member_id']:m for m in resolved['resolved_sources']}
    cache={};refs=[]
    def read(path,signed=False):
        if path not in cache:
            cache[path]=prior.read(path);paths.append(path)
            if signed:prior.verify(cache[path])
        return cache[path]
    for kind in ('paired_review','negative_review'):
        r=read(Path(design['evaluation'][kind]),True)
        for f in r['frames']:
            cp=Path(f['image_path']).parents[1]/'collection-receipt.json'
            vs=[v for v in read(cp)['views'] if v['view_id']==f['view_id']]
            if len(vs)!=1 or vs[0]['image_sha256']!=f['image_sha256']:raise ValueError('Development receipt conflict')
            refs.append(dict(view_id=f['view_id'],receipt=str(cp),pose=vs[0]['actual_pose']))
    rows=[]
    for m in p['members']:
        if not (re.fullmatch(r'[GS]\d\d-(original|warm|cool|gray_target_body|gray_all_body)',m['member_id']) or m['member_id']=='C01-original' or m['member_id'] in light):continue
        row=dict(member_id=m['member_id'],lineage_id=m['lineage_id'],gaps=[])
        try:
            for key in ('image','label'):
                path=Path(m[key+'_path'])
                if prior.file_sha256(path)!=m[key+'_sha256']:raise ValueError('Current member drift')
                paths.append(path)
            if m['member_id'] in light:
                source=light[m['member_id']];ip=Path(source['actual_render_image']);cp=Path(source['receipt']);r=read(cp,True)
                index=int(re.fullmatch(r'frame-(\d+)-rgb.png',ip.name)[1])
                frames=[v for v in r['records'] if v['capture_index']==index]
            else:
                ip=Path(m['source_image'])
                if ip.suffix=='.ppm':
                    cp=ip.parents[1]/'collection-receipt.json';r=read(cp)
                    frames=[v for v in r['views'] if v['view_id']==ip.parent.name]
                    if len(frames)!=1 or frames[0]['image_sha256']!=prior.file_sha256(ip):raise ValueError('Original frame identity conflict')
                    label_correspondence(m['truth'],frames[0]['truth']['objects'])
                else:
                    cp=ip.parents[1]/'receipt.json';r=read(cp,True)
                    index=int(re.fullmatch(r'frame-(\d+)-rgb.png',ip.name)[1])
                    frames=[v for v in r['records'] if v['frame_index']==index]
            if len(frames)!=1:raise ValueError('Frame/pose association missing or ambiguous')
            if ip.suffix!='.ppm' and r['inputs'].get(str(ip))!=prior.file_sha256(ip):raise ValueError('Replay does not bind selected RGB')
            equal_rgb(ip,m['image_path']);paths.append(ip)
            pose=frames[0]['actual_pose'];distance(pose,pose);hits=[]
            for ref in refs:
                metres,degrees=distance(pose,ref['pose'])
                if metres<=.05 and degrees<=1:hits.append(dict(view_id=ref['view_id'],receipt=ref['receipt'],metres=metres,degrees=degrees))
            row.update(status='pose_role_review_required' if hits else 'actual_source_rgb_and_pose_screen_verified',
                original_image=str(ip),actual_receipt=str(cp),actual_pose={k:pose[k] for k in ('position','orientation')},
                development_pose_matches=hits,source_role=m['data_role'])
        except (KeyError,ValueError,FileNotFoundError) as exc:
            row.update(status='source_check_blocked',gaps=[str(exc)])
        rows.append(row)
    if len(rows)!=65:raise ValueError('Unexpected final source population')
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='final_variant_sources_checked_not_dataset_ready',members=rows,
        counts=dict(Counter(m['status'] for m in rows)),
        limitation='Source-role screen only. Existing full-label review links remain required; no parent-frame pixel visibility transfer or independent-scene claim.',
        training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['counts'])
