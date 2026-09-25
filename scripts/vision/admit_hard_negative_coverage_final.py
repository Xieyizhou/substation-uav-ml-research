"""Finalize after explicit near-similarity adjudication; no automatic review."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT,read,save,verify,file_sha256
from scripts.vision.hard_negative_coverage_admission import validate_decisions

def main():
    audit_path=OUT/'final-audit.json';a=read(audit_path);verify(a)
    decisions_path=OUT/'final-decisions.json';d=read(decisions_path);verify(d)
    validate_decisions(a['frames'],d['decisions'])
    frames={r['view_id']:r for r in a['frames']};by_path={r['image_path']:r for r in a['frames']}
    near=[]
    for c in a['checks']:
        if c['exact_matches'] or c['protected_exact'] or c['protected_min_distance']<=2:raise ValueError('Exact/protected overlap')
        if c['nearest'][0]['distance']>2:continue
        r=frames[c['view_id']];peer=by_path.get(c['nearest'][0]['path'])
        if peer is None or {r['ordinal'],peer['ordinal']}!={41,42} or r['variant']!='light_cool_low' or peer['variant']!='light_cool_low':
            raise ValueError('Unreviewed near-similarity case')
        near.append(dict(view_id=r['view_id'],peer_view_id=peer['view_id'],decision='retain_same_role_linked_correlated_views',
            image_sha256=r['image_sha256'],peer_image_sha256=peer['image_sha256'],review_nature='AI-assisted',
            reason='已分别查看 remaining 064/065 与 repair 002/003：均为同一普通柜体右缘截断，杆体位置和柜体露出宽度不同。是近相似同源构图，不是独立结构证据；两对整体关联为一个相关来源簇，均仅训练候选，不拆角色。'))
    if len(near)!=2:raise ValueError('Expected exactly two directed similarity cases')
    linked=[r for r in a['frames'] if r['ordinal'] in (41,42)]
    review_path=OUT/'near-similarity-review.json'
    save(review_path,dict(status='explicitly_reviewed',decisions=near,linked_pair_ids=sorted({r['pair_id'] for r in linked}),
        correlation_cluster='medium-cabinet-edge-41-42',independent_structure_claim=False,
        inputs={r['image_path']:r['image_sha256'] for r in linked}))
    rows=[]
    for r in a['frames']:
        rows.append({**r,'correlation_group_id':'medium-cabinet-edge-41-42' if r['ordinal'] in (41,42) else r['derivation_group']})
    save(OUT/'final-admission.json',dict(status='accepted',accepted=96,frames=rows,decisions=d['decisions'],
        pose_pairs=48,correlation_clusters=47,development_training_eligible=True,
        scope='this frozen negative composition comparison only; no independent-scene claim',
        inputs={str(p):file_sha256(p) for p in (audit_path,decisions_path,review_path,OUT/'final-review-manifest.json',Path(__file__))}))
    print('ADMITTED 96 / 48 pose pairs / 47 correlation clusters; not independent scenes')

if __name__=='__main__':main()
