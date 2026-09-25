"""Explicit visual review of 46 labels, plus separate whole-image coverage gates."""
from datetime import datetime,timezone
from pathlib import Path
import re
from scripts.vision.closed_body_material_pilot import OUT,read,verify,frozen,file_sha256

EXPECTED='f570d05c3e4cceb1a171a3c12d4ae31edcf6d0b2472b6c359dc536befcec647e'
# Both warm and cool full frames and every crop were separately inspected.
# A = observed usable content for candidate consideration; H = content unresolved.
NOTES={
 'T020':[
 ('A','partial_truncation','左图缘截断，但宽阔主体、基座和上方套管可辨；不是只剩边缘条。'),
 ('A','clear_closed_body','封闭主体正面、顶部边缘及基座完整可辨；材质改变后几何保持封闭。')],
 'T036':[
 ('A','partial_occlusion','后方柜体被前方同类箱体挡住下部，顶部和上部平面可见；实例叠图确认其与前方箱体分开。'),
 ('A','partial_occlusion','普通柜体正面、顶部和基座大部分可见，右下部与邻近结构重叠。'),
 ('A','partial_occlusion','柜体正面及右侧面可辨，左部被前景杆体遮住。'),
 ('H','insufficient_or_uncertain','前景变压器遮住主要内容，实例叠图只确认后方窄顶部、侧面条带及基座边缘；实例存在不等于剩余内容足以承担类别监督，暂缓。'),
 ('A','partial_occlusion','变压器宽阔主体、套管和基座可辨，左下被前景设备遮住。'),
 ('A','partial_occlusion','变压器主体、顶部套管及基座可辨，右下被本轮变色电容器遮住。'),
 ('A','partial_occlusion','前景杆体穿过变压器，但主体、侧面和顶部套管仍可辨。'),
 ('A','partial_occlusion','后方柜体正面、顶部和左侧基座可见，右下被前景变压器遮住。'),
 ('A','partial_truncation','右图缘截断，但电抗器宽阔圆柱主体、顶面和基座仍可辨。'),
 ('A','partial_occlusion','后方柜体上部、侧面和顶部可见；实例叠图确认归属，框内套管属于前景变压器，不归给开关柜。'),
 ('A','clear_closed_body','近处封闭电容器顶部、正侧面和基座清楚可辨，未拆壳。')],
 'T023':[
 ('A','partial_occlusion','后方灰色柜体顶部、部分面板轮廓及侧面可辨，下部被本轮变色柜体遮住。'),
 ('A','partial_truncation','左图缘截断，但普通柜体侧面、顶部及基座可辨，不要求此视角显露面板。'),
 ('A','clear_body','变压器完整主体、顶部套管及基座可辨。'),
 ('A','clear_closed_body','变色柜体保持封闭主体，原灰色面板、顶部、侧面及基座可辨。'),
 ('A','partial_occlusion','电抗器圆柱主体、顶面及大部分基座可辨，右下部被变压器屋顶遮住。')],
 'T027':[
 ('A','partial_occlusion','后方柜体顶部、面板上部与右侧平面可辨，左下被前方柜体遮住。'),
 ('A','partial_truncation','近处柜体左下图缘截断，但顶部、面板轮廓和侧面可辨。'),
 ('A','partial_truncation','右图缘截断，但变压器主体、顶部套管及基座可辨。'),
 ('A','clear_closed_body','本轮变色柜体封闭主体、原面板和基座清楚可辨。'),
 ('A','partial_occlusion','电抗器圆柱主体和顶面可辨，下部被本轮变色柜体顶部遮住。')],
}


def binding(e):
    return {k:e[k] for k in ('event_id','source_truth','image_sha256','full_context_sha256','crop_sha256','receipt_identity')}


def validate(q,decisions,check_files=True):
    expected={r['event_id']:r for r in q['events']}
    if len(expected)!=len(q['events']) or len(decisions)!=len(expected) or {d['event_id'] for d in decisions}!=set(expected):raise ValueError('Missing/duplicate decision')
    for d in decisions:
        e=expected[d['event_id']]
        if d['binding']!=binding(e):raise ValueError('Stale decision binding')
        if d['decision'] not in ('observed_usable_candidate','held_content_uncertain'):raise ValueError('Unknown decision')
        if d['content']=='insufficient_or_uncertain' and d['decision']!='held_content_uncertain':raise ValueError('Unknown silently accepted')
        if d['review_nature']!='AI-assisted' or not d['reason'] or not d['reviewed_at']:raise ValueError('Missing explicit review')
        if d['training_admitted'] or d['promotable'] or d['component_pixel_certified']:raise ValueError('Unauthorized certification')
        if check_files:
            for k in ('image','full_context','crop'):
                if file_sha256(e[k+'_path'])!=e[k+'_sha256']:raise ValueError('Changed visual evidence')


def frame_gates(q,ds,mask):
    byid={d['event_id']:d for d in ds};result=[]
    for rid in ('T020','T036','T023','T027'):
        for v in ('warm','cool'):
            ev=[e for e in q['events'] if e['source_review_id']==rid and e['variant']==v]
            held=[e['event_id'] for e in ev if byid[e['event_id']]['decision']=='held_content_uncertain']
            missing=[a for a in mask['anomalies'] if a['source_review_id']==rid and a['variant']==v]
            result.append(dict(source_review_id=rid,variant=v,status='held_whole_image' if held or missing else 'reviewed_candidate_only',
                held_labels=held,unboxed_visible_instances=missing,training_admitted=False,promotable=False))
    return result


def main():
    q=read(OUT/'review-evidence.json');mask=read(OUT/'mask-coverage.json')
    for p in ('review-evidence.json','review-pages.json','mask-coverage.json'):verify(read(OUT/p))
    if q['identity']!=EXPECTED:raise ValueError('Wrong evidence for observations')
    dest=OUT/'semantic-review.json'
    if dest.exists():r=read(dest);verify(r);validate(q,r['decisions']);return
    decisions=[]
    for e in q['events']:
        idx=int(e['event_id'].rsplit('-',1)[1]);code,content,reason=NOTES[e['source_review_id']][idx]
        label=str(int(re.search(r'instance-(\d+)-',e['source_truth']['annotation_id'])[1]))
        unit=next(u for u in mask['units'] if u['source_review_id']==e['source_review_id'] and u['variant']==e['variant'])
        evidence=unit['visible_equipment'][label]
        decisions.append(dict(event_id=e['event_id'],binding=binding(e),decision='observed_usable_candidate' if code=='A' else 'held_content_uncertain',
            content=content,reason=reason,review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
            runtime_label=label,instance_evidence=evidence,component_pixel_certified=False,training_admitted=False,promotable=False))
    validate(q,decisions);frames=frame_gates(q,decisions,mask)
    paths=[OUT/p for p in ('review-evidence.json','review-pages.json','mask-coverage.json')]+[Path(__file__)]
    anomaly_observation='T027 底边 [59,1052,306,1080] 的浅灰色薄片在原样控制及两个变体均有 3534 个实例 128 像素；其颜色叠图归属 west_switchgear_02，但完整框输出缺少该实例。仅为截断片段，尚未确认应标注还是依据显式策略忽略；不得自动补标签。'
    frozen(dest,dict(status='all_46_labels_reviewed_with_whole_image_holds',decisions=decisions,frames=frames,
        anomaly_observation=dict(review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),reason=anomaly_observation),
        training_ready=False,training_started=False,original_labels_changed=False,
        inputs={str(p):file_sha256(p) for p in paths}))
    print('LABELS',len(decisions),'HELD_LABELS',sum(d['decision']=='held_content_uncertain' for d in decisions),'HELD_FRAMES',sum(f['status']=='held_whole_image' for f in frames))


if __name__=='__main__':main()
