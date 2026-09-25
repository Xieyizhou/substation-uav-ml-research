"""Read explicit decisions and quantify hypothetical impact, never change data."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.supervision_risk_revision import OUT,SOURCE,ROOT,read,verify,frozen,file_sha256,unique,IDS
from src.ml.artifacts import object_sha256

OBS={
 'T30':('evidence_pending','left','极窄图缘框内主要见前景蓝设备和上缘小片，不能认证原帧电抗器像素；重放被本地端口权限阻断，暂不提出坐标或类别新值。'),
 'T08':('evidence_pending','left','左缘可见灰色上部窄带，前景设备及浅色附件混入；映射仅由保存计划后验重建，重放未取得像素证据，保留待定。'),
 'T29':('hold_whole_image_proposed','none','原回执实例映射及完整标签一致；目标只剩前景杆和变压器之间窄灰条，可辨识内容不足。建议未来整图暂缓，不删单框；没有原帧掩码认证。'),
 'T33':('hold_whole_image_proposed','none','原回执身份一致，后方圆柱仅见上缘窄带，前景变压器和附件占框大部；建议未来整图暂缓，不用可见片段缩写 full_2d 框。'),
 'T04':('metadata_only_proposed','right','宽圆柱弧面和基座可辨、画面右缘接触；不支持删框或改类别。旧字段描述转换器裁剪行为，仅建议旁路语义说明；后验映射不是历史准入认证。'),
 'T26':('metadata_only_proposed','right','右缘圆柱顶面、宽主体和基座可辨，前景建筑角有遮挡；full_2d 框含遮挡内容符合原语义，仅建议元数据说明。'),
 'T32':('metadata_only_proposed','right','圆柱顶面及宽上身可辨，下部被变压器挡住，右缘接触；同帧历史重放证据另行重验。保留原 full_2d 框，组件归属不认证，仅建议元数据说明。'),
}

def validate(p,decisions):
    fs=unique(p['frames'],lambda r:r['event_id']);ds=unique(decisions,lambda r:r['event_id'])
    if set(ds)!=set(fs):raise ValueError('Missing/extra review')
    for eid,d in ds.items():
        f=fs[eid]
        if d.get('source_identity')!=object_sha256(f) or d.get('evidence_sha256')!=f['evidence_sha256']:raise ValueError('Stale decision identity')
        if file_sha256(f['evidence_path'])!=f['evidence_sha256']:raise ValueError('Stale review evidence')
        if not d.get('reason') or not d.get('reviewed_at') or d.get('review_nature')!='AI辅助审核':raise ValueError('Nonexplicit decision')
        if d.get('proposal') not in {'evidence_pending','hold_whole_image_proposed','metadata_only_proposed','retain','label_revision_candidate'}:raise ValueError('Unknown proposal')
        if d.get('apply_label_change') is not False or d.get('delete_box_keep_image') is not False:raise ValueError('Mutation not authorized')
        if d.get('proposal')=='label_revision_candidate' and not d.get('confirmed_conflict_evidence'):raise ValueError('No confirmed label conflict')

def review():
    p=read(OUT/'protocol.json');verify(p);inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),str(Path(__file__)):file_sha256(Path(__file__))};ds=[]
    for f in p['frames']:
        eid=f['event_id'];status,edge,reason=OBS[eid]
        attempts=list((OUT/'replay'/eid).glob('attempt-*/receipt.json'))
        for path in attempts:
            r=read(path);verify(r)
            if not r['process_cleanup_complete']:raise ValueError('Replay process not cleaned')
            inputs[str(path)]=file_sha256(path)
        ds.append(dict(event_id=eid,source_identity=object_sha256(f),evidence_sha256=f['evidence_sha256'],proposal=status,
            observed_image_boundary_contact=edge,historical_truncation_status=f['historical_truncation_status'],
            truncation_evidence_status='image_boundary_observation_not_complete_geometry_certificate',
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),reason=reason,
            original_pixel_certified=False,component_identity='unknown',apply_label_change=False,delete_box_keep_image=False))
    # A separate same-frame mask receipt can support T32 visibility, not its component identity.
    path=OUT/'reused-evidence/T32/receipt.json'
    if path.exists():
        r=read(path);verify(r)
        if r['status']!='original_pixel_evidence_certified':raise ValueError('Invalid reused evidence')
        inputs[str(path)]=file_sha256(path)
        d=next(d for d in ds if d['event_id']=='T32');d['original_pixel_certified']=True;d['same_frame_mask_evidence']=str(path)
    validate(p,ds)
    return frozen(OUT/'review.json',dict(status='explicit_proposals_not_applied',decisions=ds,
        prior_results_known_not_blind=True,new_review_panels_hide_predictions=True,inputs=inputs))

def closure(rows,selected,source_rows):
    sr={r['member_id']:r for r in source_rows};tokens={}
    for r in rows:
        t={('member',r['member_id'])}
        for k in ('derivation_group','pair_id','lineage_id'):
            if r.get(k):t.add((k,r[k]))
        s=sr[r['member_id']]
        if s.get('source_derivation_group') not in (None,'unknown'):t.add(('source_derivation',s['source_derivation_group']))
        t.add(('map_view',s['source_map'],s['source_view_id']))
        tokens[r['member_id']]=t
    result=set(selected)
    while True:
        joined=set().union(*(tokens[m] for m in result)) if result else set()
        new={m for m,t in tokens.items() if t&joined}|result
        if new==result:return sorted(result)
        result=new

def impact(rows,draws,selected):
    index=unique(rows,lambda r:r['member_id']);selected=set(selected)
    if selected-set(index) or set(draws)-set(index):raise ValueError('Unknown impact member')
    counts=Counter(draws);classes=Counter();instances=Counter()
    for mid in selected:
        classes.update(index[mid]['class_instances']);instances.update({k:v*counts[mid] for k,v in index[mid]['class_instances'].items()})
    return dict(unique_images=len(selected),full_label_instances=dict(classes),image_exposures=sum(counts[m] for m in selected),class_instance_exposures=dict(instances),
        remaining_original_exposure_count=len(draws)-sum(counts[m] for m in selected),resampling_performed=False)

def summarize():
    p=read(OUT/'protocol.json');r=read(OUT/'review.json');verify(p);verify(r);validate(p,r['decisions'])
    training=read(SOURCE/'protocol.json');src=read(SOURCE/'member-source-trace.json');reference=read(SOURCE/'review.json')
    for doc in (training,src,reference):verify(doc)
    mid={f['event_id']:f['member_id'] for f in p['frames']};dec={d['event_id']:d for d in r['decisions']}
    scenarios={'metadata_only':[], 'proposed_whole_image_hold':[mid[e] for e,d in dec.items() if d['proposal']=='hold_whole_image_proposed'],
        'pending_only':[mid[e] for e,d in dec.items() if d['proposal']=='evidence_pending'],
        'all_four_content_risks':[mid[e] for e in ('T08','T29','T30','T33')], 'all_seven_for_audit_not_recommendation':list(mid.values())}
    scenarios['registered_lineage_closure_of_four']=closure(training['rows'],scenarios['all_four_content_risks'],src['rows'])
    result={}
    for name,selected in scenarios.items():
        models={}
        for key,model in training['models'].items():
            draws=model['draws'];windows=[]
            for start in range(0,len(draws),300):windows.append(dict(step_start=start//6+1,step_end=(start+300)//6,**impact(training['rows'],draws[start:start+300],selected)))
            models[key]=dict(**impact(training['rows'],draws,selected),windows=windows)
        result[name]=dict(member_ids=selected,models=models,registered_lineages=sorted({row['lineage_id'] for row in training['rows'] if row['member_id'] in selected}),
            independence_certified=False,unknown_cross_member_relations_preserved=True)
    other=[e for e in training['reactors'] if e['event_id'] not in ('T08','T29','T30','T33')];old={d['event_id']:d for d in reference['decisions']}
    inputs={str(x):file_sha256(x) for x in (OUT/'protocol.json',OUT/'review.json',SOURCE/'protocol.json',SOURCE/'member-source-trace.json',SOURCE/'review.json',Path(__file__))}
    return frozen(OUT/'impact.json',dict(status='hypothetical_impact_only_no_pool_export',scenarios=result,
        reference_31=[dict(event_id=e['event_id'],member_id=e['member']['member_id'],content=old[e['event_id']]['identifiable_content']) for e in other],
        scope_limit='Closure uses registered relationships only; member-only identities cannot prove absence of further relatives.',inputs=inputs))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--review',action='store_true');ap.add_argument('--summarize',action='store_true');a=ap.parse_args()
    if a.review:review();print('SEVEN_DECISIONS_RECORDED')
    elif a.summarize:summarize();print('IMPACT_COMPLETE_NO_CHANGES')
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
