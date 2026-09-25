"""Compare reconstructed legacy camera poses to viewed development receipts.

This is a conservative same-pose screen, not a perceptual independence claim.
"""
import math
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,TRAIN,prior


def distance(a,b):
    pa,pb=a['position'],b['position'];qa,qb=a['orientation'],b['orientation']
    if len(pa)!=3 or len(pb)!=3 or len(qa)!=4 or len(qb)!=4:raise ValueError('Bad pose dimensions')
    if not all(math.isfinite(x) for x in pa+pb+qa+qb):raise ValueError('Invalid pose')
    na=math.sqrt(sum(x*x for x in qa));nb=math.sqrt(sum(x*x for x in qb))
    if na==0 or nb==0:raise ValueError('Zero quaternion')
    cosine=min(1.,abs(sum(x*y for x,y in zip(qa,qb))/(na*nb)))
    return math.dist(pa,pb),math.degrees(2*math.acos(cosine))


def run():
    dest=OUT/'legacy-development-pose-exclusion.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    paths=[TRAIN/'design.json',OUT/'resolved-legacy-lineages.json',OUT/'closeout-inventory-v2.json']
    design,legacy,inventory=[prior.read(p) for p in paths]
    for r in (design,legacy,inventory):prior.verify(r)
    refs=[];cache={}
    for kind in ['paired_review','negative_review']:
        path=Path(design['evaluation'][kind]);r=prior.read(path);prior.verify(r);paths.append(path)
        for f in r['frames']:
            ip=Path(f['image_path']);cp=ip.parents[1]/'collection-receipt.json'
            if cp not in cache:cache[cp]=prior.read(cp);paths.append(cp)
            views=[v for v in cache[cp]['views'] if v['view_id']==f['view_id']]
            if len(views)!=1 or views[0]['image_sha256']!=f['image_sha256']:raise ValueError('Development receipt binding conflict')
            pose=views[0]['actual_pose'];distance(pose,pose)
            refs.append(dict(role=kind,view_id=f['view_id'],receipt=str(cp),actual_pose={k:pose[k] for k in ['position','orientation']}))
    kept={m['member_id'] for m in inventory['members'] if m['stage']!='whole_frame_held'}
    rows=[];collisions=[]
    for m in legacy['members']:
        hits=[]
        for ref in refs:
            metres,degrees=distance(m['actual_pose'],ref['actual_pose'])
            if metres<=.05 and degrees<=1:
                hits.append(dict(view_id=ref['view_id'],receipt=ref['receipt'],metres=metres,degrees=degrees))
        row=dict(member_id=m['member_id'],currently_quality_supported=m['member_id'] in kept,
            source_frame_key=m['source_frame_key'],recording_key=m['recording_key'],world_sha256=m['world_sha256'],
            pose_matches=hits,source_identity_basis='post_hoc_saved_receipt_and_world_not_original_new_gate_certification')
        rows.append(row)
        if hits:collisions.append(row)
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='legacy_pose_matches_require_role_review' if collisions else 'no_legacy_pose_match_in_viewed_development_receipts',
        members=rows,development_frames=len(refs),collisions=collisions,
        screen={'position_m':.05,'orientation_degrees':1.,'purpose':'conservative role-collision flag, not collection tolerance or independent-scene threshold'},
        limitation='Only 116 reconstructed legacy members. Shared layout/assets remain; nearby different poses and nonlegacy cohorts are not cleared by this check.',
        training_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':
    r=run();print(r['status'],len(r['members']),len(r['collisions']))
