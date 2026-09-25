"""Record explicit FP observations and verify fixed historical retention gates."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.evaluate_reviewed_negative_order import OUT,KEYS,checked,compare_truth
from scripts.vision.exposure_order_retention import PRIOR
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks

# Each observation was made from this round's rendered full frame and crop.
OBS={
 0:('gray_block_body','灰色块体主体、侧面暗面板和基座；框未覆盖远处杆体主体'),
 1:('gray_block_body','冷暗条件下的灰色块体主体和侧面面板，非单独杆体'),
 2:('mixed_structure','灰色块体与前景粗杆重叠，右下含青色块体局部'),
 3:('gray_block_body','灰色块体的大面及基座，右边仅少量前景青色遮挡'),
 4:('gray_block_body','灰色块体正面暗矩形面板及边框、基座'),
 5:('gray_block_body','灰色块体大面与侧面暗矩形面板、基座'),
 6:('mixed_structure','灰色块体、前景竖杆及右下青色块体交叠'),
 7:('mixed_structure','冷暗灰色块体与前景竖杆交叠，右侧含青色块体'),
 8:('gray_block_body','灰色块体主体和基座，右缘仅贴近杆体'),
 9:('gray_block_body','冷暗灰色块体主体及侧面暗面板，右边界接近杆体'),
 10:('mixed_structure','灰色块体被粗竖杆遮挡，框含右侧青色块体局部'),
 11:('mixed_structure','灰色块体和粗竖杆重叠，右下有青色结构'),
 12:('mixed_structure','灰色块体右侧与杆体及前景青色块体共同入框'),
 13:('gray_block_body','灰色块体正面的暗矩形面板和边框，框基本覆盖主体'),
 14:('gray_block_body','近处灰色块体大面、侧面面板和基座'),
 15:('mixed_structure','灰色块体背侧大面及基座与前景粗杆重叠'),
}

def validate_review(evidence,decisions):
    expected={e['event_id']:(f,e) for f in evidence['frames'] for e in f['events']}
    if len(decisions)!=len(expected) or {d['event_id'] for d in decisions}!=set(expected):raise ValueError('Missing or duplicate review')
    for d in decisions:
        f,e=expected[d['event_id']]
        if d['prediction']!=e['prediction'] or d['image_sha256']!=f['image_sha256'] or d['evidence_sha256']!=f['evidence_sha256']:raise ValueError('Changed reviewed evidence')
        if file_sha256(f['image_path'])!=d['image_sha256'] or file_sha256(f['evidence_path'])!=d['evidence_sha256']:raise ValueError('Stale reviewed bytes')
        if not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Missing explicit decision')

def run():
    root=OUT/'error-review-v1';ep=root/'evidence.json';evidence=checked(ep);now=checked(OUT/'summary.json')
    decisions=[];stamp=datetime.now(timezone.utc).isoformat()
    for f in evidence['frames']:
        for e in f['events']:
            cat,reason=OBS[int(e['event_id'].split('-')[-1])]
            decisions.append(dict(event_id=e['event_id'],cell=e['cell'],view_id=e['view_id'],variant=e['variant'],prediction=e['prediction'],image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                content_category=cat,asset_identity='unknown_not_inferred_from_visual_shape_or_frame_subject',reason=reason,review_nature='AI辅助审核',reviewed_at=stamp,decision='reviewed_false_positive_content'))
    validate_review(evidence,decisions)
    reviewpath=root/'review.json'
    if not reviewpath.exists():write_record(reviewpath,dict(status='sixteen_prediction_crops_reviewed',decisions=decisions,unique_images=len(evidence['frames']),unique_view_ids=len({d['view_id'] for d in decisions}),
        content_counts=dict(Counter(d['content_category'] for d in decisions)),same_image_seed_counts={f['frame_id']:len({e['cell'].split('-')[-1] for e in f['events']}) for f in evidence['frames']},
        limits='Visual gray block is not a confirmed cabinet or building asset identity. Repeated seeds/lighting variants are not independent scenes.',
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in (ep,Path(__file__))}))
    reviewed=checked(reviewpath);validate_review(evidence,reviewed['decisions'])
    policy=checked(PRIOR/'protocol.json');hp=Path(policy['evaluation']['historical_reference']);hist=checked(hp)
    reference_paths=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)];refs=[checked(p) for p in reference_paths]
    histpaths=[next(Path(x) for x in hist['inputs'] if x.endswith(f'/historical-A-{s}.json')) for s in (7,17,27)];historical=[checked(p) for p in histpaths]
    if aggregate(historical)!=hist['historical_A']:raise ValueError('Historical aggregate differs')
    dependencies=[OUT/'summary.json',reviewpath,ep,PRIOR/'protocol.json',hp,*reference_paths,*histpaths,Path(__file__)]
    transitions=[]
    for seed,reference,h in zip((7,17,27),refs,historical):
        key=f'reviewed-interleaved-480-{seed}';cp=OUT/'units'/key/'completion.json';unit=checked(cp);current=checked(unit['result']);dependencies += [cp,Path(unit['result'])]
        for label,old in [('fixed_retained_reference_450',reference),('historical_A',h)]:
            by={(x['pair_id'],x['variant']):x for x in old['rows']}
            if len(by)!=48:raise ValueError('Reference membership mismatch')
            for row in current['rows']:
                states=compare_truth(by[row['pair_id'],row['variant']],row)
                if row['variant'] in ('original','lighting'):
                    transitions.append(dict(seed=seed,reference=label,pair_id=row['pair_id'],variant=row['variant'],instances=states,
                        current_misses=row['misses'],scope='Numerical transitions; not visual approval of positive labels'))
    gates={}
    for arm,g in now['groups'].items():
        result=policy_checks(g,aggregate(refs),hist['historical_A'],policy)
        for c in result['checks']:
            if c.get('reference')=='same_budget_R':c['reference']='fixed_retained_reference_450_not_same_budget_as_480'
        gates[arm]=result
    output=dict(status='fp_review_and_fixed_retention_audit_complete_no_candidate',gates=gates,retention_transitions=transitions,
        review_counts=reviewed['content_counts'],matching_conflicts=now['matching_conflicts'],selected_candidate=None,
        conclusion='Order effect is supported for this fixed reviewed data/budget. Interleaving avoids tail-collapse but fails detection and retention gates. No promotion or automatic next training.',
        next_priority='With interleaving fixed, diagnose training-side fit/coverage of gray block bodies, body-pole overlaps and clear switchgear instances before choosing any further sampling change.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in dependencies})
    dest=root/'audit.json'
    if dest.exists():return checked(dest)
    return write_record(dest,output)

if __name__=='__main__':print(run()['status'])
