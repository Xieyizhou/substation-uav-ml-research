"""Summarize explicit decisions and read-only impact; no revised training pool."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.replay_retained_bridge_remaining import OUT,PILOT,SOURCE,read,save,file_sha256,verify_tree
from scripts.vision.trace_retained_bridge_sources import ROOT,REFERENCE,baseline_verify
from scripts.vision.record_retained_bridge_review import validate
from scripts.vision.remaining_bridge_observations import expanded
from src.ml.artifacts import object_sha256

def summarize(items,trace):
    lookup={f['member_id']:f for f in trace['frames']}
    if len(items)!=159 or len({i['review_id'] for i in items})!=159:raise ValueError('Incomplete or duplicate full review')
    groups={}
    for key in sorted(trace['groups']):
        mids=trace['groups'][key];rows=[i for i in items if i['member_id'] in mids]
        expected=sum(len(lookup[mid]['objects']) for mid in mids)
        if len(rows)!=expected or len({(i['member_id'],i['runtime_label']) for i in rows})!=expected:raise ValueError('Missing full-frame instance')
        for mid in mids:
            if {(i['runtime_label'],i['object_id']) for i in rows if i['member_id']==mid}!={(o['instance_label'],o['device_id']) for o in lookup[mid]['objects']}:raise ValueError('Instance scope conflict')
        issues=[i['review_id'] for i in rows if i['review_status']!='实例可见且内容可辨识']
        groups[key]=dict(status='held_for_content_risk' if issues else 'visibility_checked_only_not_training_admitted',member_ids=mids,review_ids=[i['review_id'] for i in rows],content_risk_ids=issues,
            planned_category=lookup[mids[0]]['category'],full_instances_per_variant=dict(Counter(o['category'] for o in lookup[mids[0]]['objects'])),
            training_admitted=False,promotable=False)
    return groups

def main():
    dest=OUT/'diagnosis.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    manifest=OUT/'review-manifest.json';seen=set()
    for path in (manifest,PILOT/'decisions.json',REFERENCE/'protocol.json'):verify_tree(path,seen)
    m=read(manifest);items=m['items'];obs=expanded()
    if m['blocked_frames'] or len(items)!=135:raise ValueError('Replay or review gap')
    validate(items,obs);reviewed=[]
    for item in items:
        state,reason=obs[item['review_id']]
        reviewed.append({**item,'review_status':'实例可见且内容可辨识' if state=='identifiable' else '实例有可见证据，但内容不足',
            'review_nature':'AI辅助审核','reviewed_at':datetime.now(timezone.utc).isoformat(),'reason':reason,
            'component_evidence':'unknown','class_discriminability':'not_certified','training_admitted':False,'promotable':False})
    all_items=read(PILOT/'decisions.json')['items']+reviewed;trace=read(SOURCE/'trace.json');groups=summarize(all_items,trace)
    held={mid for g in groups.values() if g['content_risk_ids'] for mid in g['member_ids']}
    protocol=read(REFERENCE/'protocol.json');members={r['member_id']:r for r in protocol['pool_rows']};exposure={}
    for seed in (7,17,27):
        seq=protocol['schedules'][f'I-300-{seed}'];risk=[mid for mid in seq if mid in held]
        counts=Counter()
        for mid in risk:counts.update(members[mid]['class_instances'])
        exposure[str(seed)]=dict(hypothetical_whole_group_removed_draws=len(risk),hypothetical_removed_full_instance_exposures=dict(counts),reweighted=False)
    replay_rows=[]
    for folder in (PILOT,OUT):
        for f in read(folder/'progress.json')['frames']:
            r=read(f['receipt_path'])
            if r['status']!='original_pixel_evidence_certified' or not r['process_cleanup_complete']:raise ValueError('Replay incomplete or cleanup missing')
            replay_rows.extend(r['records'])
    paths=[manifest,PILOT/'decisions.json',SOURCE/'trace.json',REFERENCE/'protocol.json',Path(__file__),ROOT/'scripts/vision/remaining_bridge_observations.py',ROOT/'scripts/vision/record_retained_bridge_review.py']
    save(dest,dict(status='all_retained_bridge_visibility_diagnosis_complete_not_training_admission',remaining_decisions=reviewed,groups=groups,
        total_frames=39,total_lineages=13,total_boxes=159,remaining_batch_counts=dict(Counter(i['review_status'] for i in reviewed)),all_counts=dict(Counter(i['review_status'] for i in all_items)),
        group_status_counts=dict(Counter(g['status'] for g in groups.values())),whole_group_hypothetical_removal_impact=exposure,
        alignment=dict(certified_frames=39,maximum_skew_ms=max(r['skew_ms'] for r in replay_rows),maximum_box_delta_px=max(r['maximum_box_delta_px'] for r in replay_rows),all_rgb_exact=all(r['rgb_exact'] for r in replay_rows),all_cleanup_complete=True),
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        next_action='Design a separate full-image-preserving revision with explicit exposure/scale/lineage accounting; no automatic label deletion or training.',
        limitations=['Visibility checked is not proof of unique class geometry or generalization.',
            'Same-source variants and repeated exposures are not independent scenes.',
            'Full_2d occluded extents alone do not prove label corruption.',
            'Only the 39 retained bridge members were fully replayed here; other training subsets are outside this visibility audit.'],
        inputs={str(p):file_sha256(p) for p in paths}))
    print('DIAGNOSIS_COMPLETE',Counter(g['status'] for g in groups.values()),Counter(i['review_status'] for i in all_items),flush=True)

if __name__=='__main__':main()
