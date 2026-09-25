"""Explicit observations after individually inspecting ten certified mask pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.replay_development_visibility_gaps import OUT, prior

OBSERVATIONS = {
    'LOSS03': '灯光条件：目标只露出后排最远柜体的顶面与上沿窄带；框内下方宽蓝色面属于前景柜体。目标实例存在，但缺少完整主体和可辨面板。',
    'LOSS12': '灯光条件：右缘后排目标露出倒L形顶面及窄侧条，前景邻柜遮住主体大部；可确认所属实例，不能仅凭该片段独立确认设备类别。',
    'LOSS13': '灯光条件：右缘第二层柜体露出较宽竖向侧条、顶沿和底部小片段；前景柜体占据框内右下部，完整主体及面板不可辨。',
    'LOSS16': '原始条件：右缘后排目标的顶沿、左侧窄条与底部小片段可归属，主体被邻柜遮挡并触及图缘；类别辨识信息有限。',
    'LOSS17': '原始条件：右缘中层柜体的竖向侧条与顶沿可见，右下大片蓝色区域属于前景邻柜；不能把整框内容视为目标主体。',
    'LOSS20': '灯光条件：变压器仅在最左图缘露出窄小上部块体，下面的面板和宽蓝色区域属于前景开关柜；端子及完整主体不可见，内容不足。',
    'LOSS23': '灯光条件：右后方目标仅露出顶面与窄上沿，下面两层蓝色面属于更近柜体；身份已解开，但面板与完整主体仍不可辨。',
    'LOSS29': '原始条件：右后方目标的顶面和上沿条带可见，主体被同排前景柜体遮挡；不因同位姿另一条件已核验而代替本帧审核。',
    'LOSS38': '灯光条件：左图缘后方柜体仅剩顶部与上侧片段，前景柜体遮住下部且目标延伸至画缘；没有足够完整外形。',
    'LOSS40': '原始条件：左图缘后方目标仅露出顶面及短侧面片段，框中大块蓝色面属于前景柜体；像素归属明确但内容不足。',
}


def validate(evidence, decisions):
    events = {x['event_id']: x for x in evidence['events']}
    if len(events) != len(evidence['events']):
        raise ValueError('Duplicate evidence')
    if len(decisions) != len(events) or {d['event_id'] for d in decisions} != set(events):
        raise ValueError('Missing or duplicate decision')
    for d in decisions:
        e = events[d['event_id']]
        if any(d[k] != e[k] for k in ('image_sha256', 'page_sha256', 'object_id', 'runtime_label', 'truth')):
            raise ValueError('Stale evidence or identity')
        if not e['original_pixel_evidence_certified'] or e['visible_pixels'] <= 0:
            raise ValueError('Positive original pixel evidence required')
        if d['status'] != 'visible_but_insufficient_content' or not d['reason'] or not d['reviewed_at'] or d['review_nature'] != 'AI-assisted':
            raise ValueError('Invalid explicit decision')
        if d['training_admitted'] or d['promotable']:
            raise ValueError('Admission forbidden')


def main():
    ep = OUT / 'evidence.json'
    e = prior.read(ep)
    prior.verify(e)
    if set(OBSERVATIONS) != {x['event_id'] for x in e['events']}:
        raise ValueError('Explicit observation coverage changed')
    now = datetime.now(timezone.utc).isoformat()
    decisions = [dict(event_id=x['event_id'], status='visible_but_insufficient_content',
                      reason=OBSERVATIONS[x['event_id']], reviewed_at=now, review_nature='AI-assisted',
                      component_evidence='unknown', visual_component_description='partial top/side surface; not component-segmented',
                      training_admitted=False, promotable=False,
                      **{k: x[k] for k in ('image_sha256', 'page_sha256', 'object_id', 'runtime_label', 'truth')})
                 for x in e['events']]
    validate(e, decisions)
    prior.frozen(OUT / 'review.json', dict(status='ten_explicit_visibility_decisions_recorded', decisions=decisions,
        scope='Pixel attribution resolved; content insufficiency is an AI-assisted descriptive judgment, not a validated area threshold or label correction.',
        inputs={str(p): prior.file_sha256(p) for p in (ep, Path(__file__).resolve())}))
    print('10 explicit decisions; no admission and no label revisions')


if __name__ == '__main__':
    main()
