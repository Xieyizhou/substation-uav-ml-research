"""Import explicit visual observations; never synthesize missing reviews."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.interleaved_small_scale_control import OUT, prior
from scripts.vision.record_small_scale_fp_review import validate as validate_fp
from scripts.vision import interleaved_error_observations as notes
from scripts.vision.exposure_order_retention import baseline_verify


def validate_loss(evidence, observations):
    objects = evidence['objects']
    ids = [o['review_id'] for o in objects]
    if len(set(ids)) != len(ids) or set(ids) != set(observations):
        raise ValueError('Missing, extra or duplicate loss review')
    decisions = []
    for o in objects:
        category, reason = observations[o['review_id']]
        if category not in {'clear_body','partial_occlusion','edge_truncated','limited_content','unknown'} or not reason.strip():
            raise ValueError('Invalid explicit observation')
        decisions.append(dict(review_id=o['review_id'], source=o['source'], truth=o['truth'],
            events=o['events'], page_path=o['page_path'], visual_content=category, reason=reason,
            status='pending' if category=='unknown' else 'visual_content_described',
            pixel_visibility_certified=False, review_nature='AI辅助审核'))
    if len(decisions)!=evidence['unique_loss_targets'] or sum(len(d['events']) for d in decisions)!=evidence['new_loss_events']:
        raise ValueError('Loss coverage count mismatch')
    return decisions


def run():
    fp=OUT/'evaluation/error-review-v1/evidence.json'
    loss=OUT/'evaluation/loss-review-v1/evidence.json'
    summary=OUT/'evaluation/summary.json'
    deps=[fp,loss,summary,OUT/'evaluation/error-review-v1/adapter-binding.json',Path(notes.__file__).resolve(),Path(__file__).resolve()]
    records=[prior.read(p) for p in deps[:4]]
    for r in records: prior.verify(r)
    f,l,s,_=records
    # Validate the evidence's evaluation inputs as well as their immediate hashes.
    for r in (f,l):
        for p in r['inputs']:
            if Path(p).suffix=='.json':
                v=prior.read(p)
                if 'identity' in v and 'inputs' in v: prior.verify(v)
    fd=validate_fp(f,notes.FP); ld=validate_loss(l,notes.LOSS)
    now=datetime.now(timezone.utc).isoformat()
    for d in fd+ld:
        d['recorded_at']=now
        d['page_sha256']=prior.file_sha256(d['page_path'])
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    deps.append(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    pending=[d.get('review_id',d.get('event_id')) for d in fd+ld if d['status']=='pending']
    result=dict(status='review_pending' if pending else 'error_review_complete_model_gates_failed',
        recorded_at=now,selected_candidate=None,review_complete=not pending,pending=pending,
        false_positive_decisions=fd,loss_decisions=ld,
        fp_counts=dict(Counter(d['visual_structure'] for d in fd)),
        loss_content_counts=dict(Counter(d['visual_content'] for d in ld)),
        loss_reasons=dict(Counter(e['miss']['reason'] for d in ld for e in d['events'])),
        loss_unique_images=len({d['source']['image_sha256'] for d in ld}),
        transition_counts=l['transition_counts'],
        failed_gates={k:[c for c in v['checks'] if not c['passed']] for k,v in s['policies'].items()},
        matching_conflicts=s['matching_conflicts'],baseline=b,
        limitations=['AI视觉描述不是实例掩码认证；图缘观察不修改历史截断标签。',
            '低阈值漏检类型为机器匹配诊断，未声称逐一视觉审核全部低阈值预测。',
            '29预测事件与64漏检事件含重复seed；不是独立样本数。',
            '逐图范围为全部无目标误检及原始/光照新增漏检；其余转移保留数值记录。'],
        inputs={str(p):prior.file_sha256(p) for p in deps})
    if any(v['passed'] for v in s['policies'].values()):
        raise ValueError('Unexpected policy change; reconsider completion wording')
    dest=OUT/'evaluation/error-review-v1/review.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,result)


if __name__=='__main__':
    r=run();print(r['status'],r['fp_counts'],r['loss_content_counts'],r['loss_reasons'])
