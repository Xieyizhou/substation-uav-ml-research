"""Second-round explicit decisions and fail-closed stop receipt, no training."""
from datetime import datetime,timezone
from collections import Counter
from pathlib import Path
import subprocess,sys
from scripts.vision.risk_capped_second_review import OUT as SECOND,CONTROL,ROOT,ready,read,verify,frozen,file_sha256
from scripts.vision.decide_risk_capped_control import validate
from scripts.vision.check_risk_capped_redistribution import validate_capped
from scripts.vision.prepare_whole_image_hold import baseline_verify
from scripts.vision.structure_fit import pixels

OBS={
'N01':['独立完整块体顶面侧面与底座'],
'N02':['右图缘宽顶面侧面与单个柱附件可辨，近景截断','完整独立块体及底座'],
'N03':['主体三柱可辨，右下局部遮挡','完整宽主体顶面及底座'],
'N04':['主体三柱底座可辨，右图缘截断','完整侧面顶面及底座，面板未见','后排宽侧面底座可辨，右侧前景遮挡','宽主体顶面底座可辨，左图缘截断'],
'N05':['后排面板轮廓部分可见，下部遮挡','宽面板和侧面可辨，左下图缘截断','完整主体三柱和底座','完整灰色面板主体底座','圆柱主体可辨，下部局部被前景遮挡'],
'N06':['后方面板可辨，前景遮挡下部','宽侧壁顶面及底座，左图缘截断','完整主体三柱与底座','完整面板侧面和底座','圆柱大部曲面可辨，右下局部遮挡'],
'N07':['!前景后仅重叠的上侧平面，面板及整体轮廓不足','主体侧壁和底座可辨，杆遮中央','!前景间仅一段无面板蓝色平面，目标可辨内容不足','圆柱大部曲面及底座可辨，左侧局部遮挡'],
'N08':['后方面板大部可辨，下部遮挡','灰蓝色宽侧壁顶面及底座，左缘截断','完整主体三柱与底座','完整面板侧面及底座','圆柱主体可辨，右下被前景遮挡'],
'N09':['完整宽主体、顶面三柱及底座'],
}

def require_quality_ready(first,second,resolve):
    if resolve['additional_resolves_used']!=1 or resolve['additional_resolves_remaining']!=0:raise ValueError('Invalid resolve budget')
    if not first['decisions'] or not second['decisions']:raise ValueError('Missing review')
    if any(d['status']=='insufficient_or_uncertain' for d in second['decisions']):raise ValueError('New risk after single permitted resolve: training blocked')

def main():
    e=read(SECOND/'evidence.json');verify(e);ds=[]
    if set(OBS)!={f['event_id'] for f in e['events']}:raise ValueError('Observation scope mismatch')
    for f in e['events']:
        if len(OBS[f['event_id']])!=len(f['labels']):raise ValueError('Missing box observation')
        for l,note in zip(f['labels'],OBS[f['event_id']]):
            ds.append(dict(event_id=l['event_id'],member_id=f['member']['member_id'],truth=l['truth'],object_id=l['object_id'],
                image_sha256=f['member']['image_sha256'],label_sha256=f['member']['label_sha256'],crop_sha256=l['crop_sha256'],
                status='insufficient_or_uncertain' if note.startswith('!') else 'identifiable_geometry_with_recorded_limits',
                reason=note.lstrip('!'),review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
                pixel_visibility_certified=False,training_approved=False))
    validate(e,ds)
    review=frozen(SECOND/'review.json',dict(status='second_round_new_risk_training_blocked',decisions=ds,
        inputs={str(x):file_sha256(x) for x in [SECOND/'evidence.json',Path(__file__)]}))
    first=read(CONTROL/'review.json');initial=read(CONTROL/'evidence.json');r=read(CONTROL/'resolve-01.json');p=ready()
    for d in (first,initial,r):verify(d)
    validate(initial,first['decisions'])
    try:require_quality_ready(first,review,r)
    except ValueError as ex:blocked_reason=str(ex)
    else:raise ValueError('Expected explicit stop condition absent; do not publish blocked receipt')
    a=next(f for f in initial['events'] if f['event_id']=='C22');b=next(f for f in e['events'] if f['event_id']=='N07')
    pose_equal=all(a['actual_pose'][k]==b['actual_pose'][k] for k in ('position','orientation'))
    labels_equal=[(x['object_id'],x['truth']['bbox_xyxy']) for x in a['labels']]==[(x['object_id'],x['truth']['bbox_xyxy']) for x in b['labels']]
    same_lineage=a['member']['lineage_id']==b['member']['lineage_id'];rgb_equal=pixels(a['source_image'])==pixels(b['source_image'])
    if not (pose_equal and labels_equal and same_lineage) or rgb_equal:raise ValueError('Reported sibling evidence mismatch')
    exposure={}
    for seed,x in r['results'].items():
        old=Counter(p['schedules'][f'reference-450-{seed}']);validate_capped(p['pool_rows'],old,x['counts'],set(p['held_member_ids']),set(r['risk_members']))
        exposure[seed]={f['event_id']:dict(before=old[f['member']['member_id']],after=x['counts'][f['member']['member_id']]) for f in (a,b)}
    tests=['tests.test_risk_capped_control_review','tests.test_risk_capped_redistribution','tests.test_positive_redistribution','tests.test_redistribution_target_attribution','tests.test_redistribution_occlusion_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    for name in ('ready.json','training','preflight','sampling.json'):
        if (CONTROL/name).exists():raise ValueError('Unexpected downstream artifact despite quality stop')
    paths=[CONTROL/'review.json',CONTROL/'resolve-01.json',SECOND/'review.json',Path(__file__),ROOT/'docs/results/ml_risk_capped_control_stop_20260909.md',ROOT/'config/perception/visual_experiment_baseline_v1.json']
    paths.extend(ROOT/(t.replace('.','/')+'.py') for t in tests)
    frozen(CONTROL/'completion.json',dict(status='stopped_at_second_round_quality_gate_no_training',reason=blocked_reason,
        reviewed_images=49,reviewed_labels=184,first_round_risk_images=len(first['new_risk_members']),second_round_risk_events=[d['event_id'] for d in ds if d['status']=='insufficient_or_uncertain'],
        sibling_evidence=dict(first_event='C22',second_event='N07',same_registered_lineage=same_lineage,actual_pose_equal=pose_equal,instance_and_box_coordinates_equal=labels_equal,RGB_equal=rgb_equal,exposure=exposure),
        additional_resolves_used=1,new_training_units=0,new_evaluations=0,sequence_generated=False,readiness_issued=False,
        training_started=False,labels_modified=False,baseline=baseline,regression_output=result.stderr,whole_repository_tested=False,
        next_direction='Independent lineage-aware risk inventory and data-policy revision proposal; no further solve or training authorized in this bounded stage.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print('STOPPED_SECOND_ROUND_RISK; REVIEWED49_IMAGES184_LABELS; PINNED40_PASSED; NO_TRAINING')

if __name__=='__main__':main()
