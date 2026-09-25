"""Record source-frame and recording relations, without inferring independence."""
from collections import defaultdict
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    tp = OUT/'legacy-world-source-trace.json'
    qp = OUT/'quality-isolation-v3.json'
    trace, quarantine = prior.read(tp), prior.read(qp)
    prior.verify(trace); prior.verify(quarantine)
    frames, recordings = defaultdict(list), defaultdict(list)
    rows = []
    for member in trace['members']:
        recording = member['source_recording']
        frame = recording + '/capture/' + member['source_view_id']
        frames[frame].append(member['member_id'])
        recordings[recording].append(member['member_id'])
        rows.append(dict(member_id=member['member_id'], source_frame_key=frame,
            recording_key=recording, world_sha256=member['world_sha256'],
            source_view_id=member['source_view_id'], actual_pose=member['actual_pose'],
            interpretation='Source identity reconstructed post hoc; no original-gate or independent-scene claim.'))
    risks = quarantine['unresolved_risk_lineage_members']
    findings = []
    for mid in risks:
        matches = [r for r in rows if r['member_id'] == mid]
        if len(matches) != 1: raise ValueError('Risk source not unique')
        row = matches[0]
        findings.append(dict(member_id=mid,
            same_source_frame_members=frames[row['source_frame_key']],
            same_recording_members=recordings[row['recording_key']],
            scope='116 traced legacy members only; remaining pool derivations require separate validation.',
            whole_recording_automatically_quarantined=False,
            source_frame_resolved=True, pool_wide_derived_relationships_resolved=False))
    dest = OUT/'resolved-legacy-lineages.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    paths = [tp, qp, Path(__file__).resolve()]
    return prior.frozen(dest, dict(status='legacy_source_relations_resolved_partial_pool_scope',
        members=rows, source_frame_count=len(frames), recording_count=len(recordings),
        risk_impact=findings, dataset_ready=False, training_started=False,
        inputs={str(p): prior.file_sha256(p) for p in paths}))


if __name__ == '__main__':
    r = run(); print(len(r['members']), r['source_frame_count'], r['recording_count'])
