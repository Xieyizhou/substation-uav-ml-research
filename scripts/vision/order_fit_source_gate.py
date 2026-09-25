"""Aggregate verified source screens; never infer whole-frame quality from them."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior

SOURCES={
 'legacy-development-pose-exclusion.json':None,
 'explicit-cohort-source-role-check.json':'source_pixels_full_labels_mapping_and_pose_screen_verified',
 'negative-source-role-check.json':'original_negative_source_and_pose_screen_verified',
 'variant-source-role-check.json':'actual_source_rgb_and_pose_screen_verified',
}


def run():
    dest=OUT/'source-screen-completion.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    qp=OUT/'closeout-inventory-v2.json';q=prior.read(qp);prior.verify(q);paths=[qp];links={}
    for name,wanted in SOURCES.items():
        path=OUT/name;r=prior.read(path);prior.verify(r);paths.append(path)
        for m in r['members']:
            mid=m['member_id']
            if mid in links:raise ValueError('Ambiguous source-screen binding')
            if wanted is None:
                passed=not m['pose_matches']
            else:passed=m['status']==wanted and not m.get('development_pose_matches',[])
            links[mid]=dict(evidence=str(path),passed=passed)
    selected=[m for m in q['members'] if m['stage']=='quality_supported_source_role_gate_pending']
    rows=[dict(member_id=m['member_id'],source_screen=links.get(m['member_id']),
        status='source_screen_verified' if links.get(m['member_id'],{}).get('passed') else 'source_screen_missing_or_conflicted',
        full_frame_quality_inferred=False,training_eligible=False) for m in selected]
    missing=[m for m in rows if m['status']!='source_screen_verified']
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='all_candidate_source_screens_verified' if not missing else 'source_screen_incomplete',
        members=rows,checked_candidates=len(rows),missing_or_conflicted=missing,
        interpretation='Source identity and bounded viewed-development pose exclusion only; exact file/pixel and protected fingerprint checks remain separately bound. Shared layouts/assets remain. Full-frame approval and training schedule are not inferred.',
        dataset_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':
    r=run();print(r['status'],r['checked_candidates'],len(r['missing_or_conflicted']))
