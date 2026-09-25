"""Explicit observations of the four displayed residual fitting pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_routed_amplitude_fit import OUT, base
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES = {
    'S02-cool': '左侧变压器顶面三柱、两个大侧面及基座清楚，无主要遮挡或图缘截断；不是仅局部内容。',
    'S02-warm': '与冷色变体同源视角，左侧变压器三柱、顶面、主体和基座清楚；不能把两个变体算成独立场景。',
    'S07-warm': '近处变压器顶面三柱、大正面、侧面与底座可辨，未见主要前景遮挡，目标占图较大。',
    'small-material-v1-S14': '远处灰柜顶面、大侧背面和基座可见，右侧部分被前景变压器遮挡；尺度较小，无可见正面面板。',
}


def run():
    ep = OUT / 'residual-review-v1/evidence.json'
    evidence = base.checked(ep)
    if {f['member_id'] for f in evidence['frames']} != set(NOTES):
        raise ValueError('Residual review coverage changed')
    stamp = datetime.now(timezone.utc).isoformat()
    decisions = []
    for frame in evidence['frames']:
        for event in frame['events']:
            decisions.append(dict(
                member_id=frame['member_id'], event=event,
                reason=NOTES[frame['member_id']], review_nature='AI辅助审核',
                reviewed_at=stamp, image_sha256=frame['image_sha256'],
                label_sha256=frame['label_sha256'], evidence_sha256=frame['page_sha256'],
                pixel_visibility_certified=False,
                decision='visible_content_observed_not_training_admission'))
    return write_record(OUT / 'residual-review-v1/review.json', dict(
        status='four_residual_events_visually_reviewed', decisions=decisions,
        limits='Only residual training-fitting misses reviewed here; development-error and training false-positive reviews remain separate.',
        training_admitted=False, promotable=False,
        inputs={str(p.resolve()): file_sha256(p) for p in (ep, Path(__file__))}))


if __name__ == '__main__':
    print(run()['status'])
