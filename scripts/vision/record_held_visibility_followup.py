"""Explicit AI-assisted follow-up observations; no admission or relabeling."""
from datetime import datetime, timezone
from collections import Counter
from scripts.vision.build_held_visibility_followup import OUT, SOURCE, read, save, file_sha256, verify_tree, Path

OBS = {
'T049':('edge_depth_supported','右边缘主要为地面及少量深色结构；既有深度核验支持基座和极窄主体，不能认作空框。'),
'T054':('edge_depth_supported','左边缘主要为地面网格；既有投影及深度证据支持仅基座进入画面。'),
'T055':('edge_fragment','右边缘仅剩青色箱体窄条与顶部边缘；不足以确认面板及实例可辨识性。'),
'T077':('overlap_assignment','框内青色顶面、窄立面及右侧暗色小片相接；面板与实例归属不清。'),
'T095':('foreground_occlusion','框内前景大顶面和立柱占据主要区域，后方仅露青色顶部窄带。'),
'T111':('foreground_occlusion','前景箱体顶面及立柱遮住下部，后方可见短侧面，实例可见范围未确认。'),
'T116':('foreground_occlusion','前景深色箱体遮住大部分框，后方只露青色上沿及侧面。'),
'T117':('foreground_occlusion','前景顶面和立柱覆盖框内主要区域，后方青色细条不足以独立辨识。'),
'T124':('edge_fragment','右边界窄框内主要是深青色平面，缺少独立目标轮廓。'),
'T131':('overlap_assignment','立杆和前方带暗面板箱体叠在框中；后方箱体露出顶部，不能把前方面板归给目标。'),
'T148':('edge_depth_supported','右边缘地面网格为主；既有投影及深度证据支持仅基座进入画面。'),
'T159':('edge_fragment','左边缘可见斜侧面、基部及邻近箱体片段，截断与重叠使归属不清。'),
'T175':('overlap_assignment','前景深色箱体遮挡下部，左侧暗色片与后方青色箱体重叠，面板归属不明确。'),
'T178':('overlap_assignment','可见顶部及青色立面，左侧暗色片与前景平面相接；有箱体证据但不能确认具体面板归属。'),
'T187':('foreground_occlusion','前景大顶面和立柱遮挡，左上仅剩后方青色短侧面。'),
'T236':('foreground_occlusion','框内主要是前景顶面及三根立柱，后方仅见上缘窄带。'),
'T241':('foreground_occlusion','灰色条件下前景顶面和立柱占据框内，后方轮廓更难区分。'),
'T246':('foreground_occlusion','前景顶面与三根立柱遮挡目标，仅有后方窄带可见。'),
'T269':('foreground_occlusion','框内大部分为前景箱体顶面及立面，后方青色上缘很窄。'),
'T270':('foreground_occlusion','前景顶面及右侧立柱覆盖框，后方只有青色窄条和小暗片。'),
'T274':('foreground_occlusion','灰色前景箱体遮挡后方，框上沿仅剩薄轮廓。'),
'T275':('foreground_occlusion','灰色顶面及立柱遮挡，后方细条与小暗片不能可靠归属。'),
'T279':('foreground_occlusion','前景箱体顶面和侧面占据主要框内区域，后方只露上缘。'),
'T280':('foreground_occlusion','前景顶面及右侧立柱遮挡，后方青色细条未形成独立轮廓。'),
'T284':('foreground_occlusion','前景大箱体及两根立柱覆盖框内主要部分，后方仅有青色上缘。'),
'T288':('foreground_occlusion','灰色前景大箱体覆盖框内，后方仅剩顶部薄条；右侧邻近面板不能归给该框。'),
'T292':('foreground_occlusion','前景箱体与两根立柱遮挡，后方青色窄条不足以确认完整可见区域。'),
}

def validate(manifest, decisions):
    expected={r['review_id']:r for r in manifest['items']}
    if len(decisions)!=len(expected) or {r['review_id'] for r in decisions}!=set(expected):
        raise ValueError('Missing or duplicate review')
    for d in decisions:
        r=expected[d['review_id']]
        if d['decision']!='held_pending_instance_visibility' or d['review_nature']!='AI-assisted' or not d['reason']:
            raise ValueError('Unverified admission')
        for k in ('image','label'):
            if file_sha256(r[k+'_path'])!=d[k+'_sha256']:raise ValueError('Stale source')
        if file_sha256(r['followup_evidence_path'])!=d['evidence_sha256']:raise ValueError('Stale evidence')

def main():
    mp=OUT/'manifest.json';verify_tree(mp);m=read(mp)
    if set(OBS)!={r['review_id'] for r in m['items']}:raise ValueError('Observation coverage mismatch')
    decisions=[]
    for r in m['items']:
        category,reason=OBS[r['review_id']]
        decisions.append(dict(review_id=r['review_id'],member_id=r['member_id'],lineage_id=r['lineage_id'],
            category=category,reason=reason,decision='held_pending_instance_visibility',review_nature='AI-assisted',
            reviewed_at=datetime.now(timezone.utc).isoformat(),image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],
            evidence_sha256=file_sha256(r['followup_evidence_path']),bbox_xyxy=r['bbox_xyxy']))
    validate(m,decisions)
    members={r['member_id'] for r in m['items']};lineages={r['lineage_id'] for r in m['items']}
    allrows=read(SOURCE/'manifest.json')['items']
    closure=sorted({r['member_id'] for r in allrows if r['lineage_id'] in lineages})
    stats=dict(events=len(decisions),unique_members=len(members),unique_lineages=len(lineages),
               categories=dict(Counter(d['category'] for d in decisions)),
               lineage_closure_members_in_switchgear_manifest=len(closure))
    save(OUT/'review.json',dict(status='followup_complete_visibility_unresolved',decisions=decisions,counts=stats,
        suggested_hold_members=sorted(members),lineage_closure_members=closure,
        closure_scope='Only current switchgear manifest; not an exhaustive full-pool lineage closure.',
        historical_admission_changed=False,training_admitted=False,promotable=False,
        next_action='Instance-mask replay and provenance verification before any data revision; do not delete boxes.',
        inputs={str(p):file_sha256(p) for p in (mp,Path(__file__),SOURCE/'edge-source-audit-v1/depth-support-v1/audit.json')}))
    print(stats)

if __name__=='__main__':main()
