"""Resolve reviewed negative frames to original capture receipts, not export paths."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,TRAIN,prior
from scripts.vision.order_fit_legacy_pose_exclusion import distance
from scripts.vision.check_structure_fit_sources import equal_rgb


def run():
    dest=OUT/'negative-source-role-check.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[OUT/'protocol.json',OUT/'negative-review-links.json',TRAIN/'design.json']
    p,links,design=[prior.read(path) for path in paths]
    for r in (p,links,design):prior.verify(r)
    pool={m['member_id']:m for m in p['members']};cache={};refs=[]
    def read_receipt(path):
        if path not in cache:cache[path]=prior.read(path);paths.append(path)
        return cache[path]
    for kind in ('paired_review','negative_review'):
        path=Path(design['evaluation'][kind]);review=prior.read(path);prior.verify(review);paths.append(path)
        for f in review['frames']:
            cp=Path(f['image_path']).parents[1]/'collection-receipt.json'
            vs=[v for v in read_receipt(cp)['views'] if v['view_id']==f['view_id']]
            if len(vs)!=1 or vs[0]['image_sha256']!=f['image_sha256']:raise ValueError('Development identity conflict')
            refs.append(dict(view_id=f['view_id'],receipt=str(cp),pose=vs[0]['actual_pose']))
    rows=[]
    for link in links['members']:
        m=pool[link['member_id']];d=link['decision'];row=dict(member_id=m['member_id'],gaps=[])
        try:
            ip=Path(link.get('source_image') or d['image_path']);cp=ip.parents[1]/'collection-receipt.json'
            receipt=read_receipt(cp);vs=[v for v in receipt['views'] if v['view_id']==ip.parent.name]
            if len(vs)!=1:raise ValueError('Ambiguous source frame')
            v=vs[0]
            if prior.file_sha256(ip)!=v['image_sha256'] or d['image_sha256']!=v['image_sha256']:raise ValueError('Source/review hash conflict')
            equal_rgb(ip,m['image_path'])
            if v['truth']['objects'] or m['truth'] or Path(m['label_path']).read_text().strip():raise ValueError('Nonempty negative truth')
            if d.get('decision')!='accepted' or not d.get('reason'):raise ValueError('Missing negative decision')
            if receipt.get('actual_annotation_mode')!='full_2d':raise ValueError('Negative box mode unresolved')
            pose=v['actual_pose'];distance(pose,pose);hits=[]
            for ref in refs:
                metres,degrees=distance(pose,ref['pose'])
                if metres<=.05 and degrees<=1:hits.append(dict(view_id=ref['view_id'],receipt=ref['receipt'],metres=metres,degrees=degrees))
            row.update(status='pose_role_review_required' if hits else 'original_negative_source_and_pose_screen_verified',
                original_image=str(ip),source_receipt=str(cp),view_id=v['view_id'],derivation_group=link['derivation_group'],
                actual_pose={k:pose[k] for k in ('position','orientation')},development_pose_matches=hits,
                source_world_sha256=receipt.get('world_sha256'),retained_review=d)
            paths.extend([ip,Path(m['image_path']),Path(m['label_path'])])
        except (ValueError,KeyError,FileNotFoundError) as exc:
            row.update(status='source_check_blocked',gaps=[str(exc)])
        rows.append(row)
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='negative_sources_checked_not_dataset_admission',members=rows,
        counts=dict(Counter(m['status'] for m in rows)),
        limits='Retains existing individual no-target review, not an automatic new visual pass. Receipt world hash is recorded, not re-certified against world geometry here. Same-pose screen is not independence certification.',
        training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['counts'])
