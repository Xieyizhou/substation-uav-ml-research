"""Explicit decisions after viewing all four pairs and eleven target crops."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.unified_hold_lighting_counts import OUT,prior

NOTES={
('L01','208'):'光照后圆柱及平台变暗偏冷，边界和主体仍清楚，无新增遮挡。',
('L01','57'):'后方右段主体与顶部附件仍可辨，左侧既有遮挡保留；掩码区分前后同类。',
('L01','64'):'右部主体、顶部附件及底部仍可辨，左侧被圆柱遮挡，光照未改变此关系。',
('L02','113'):'柜体长侧面及平台仍可辨，未见前面板；暗化后轮廓没有消失。',
('L02','208'):'圆柱主体与平台完整清楚，亮度变化未新增内容不足。',
('L02','57'):'后方右段主体及附件仍可辨，左侧同类前景遮挡不变。',
('L02','64'):'右部主体、两柱及底部仍可辨，保留圆柱遮挡限制。',
('L02','76'):'后方只见上部主体带和三个附件，仍可辨但严重遮挡；未把前方柜体纳入该实例。',
('L03','208'):'圆柱和平台清晰，变暗后仍有可辨边界，没有新增截断。',
('L03','64'):'右缘截断不变，大片主体侧面、顶部附件及平台仍可辨；不声称完整可见。',
('L04','208'):'顶部、侧面与平台仍清楚，偏冷暗化未破坏可辨内容。',
}

def validate(e,decisions):
    expected={(f['pair_id'],t['label']) for f in e['events'] for t in f['targets']};keys=[(d['pair_id'],d['label']) for d in decisions]
    if set(keys)!=expected or len(keys)!=len(set(keys)):raise ValueError('Missing or duplicate review')
    for f in e['events']:
        for t in f['targets']:
            d=next(d for d in decisions if (d['pair_id'],d['label'])==(f['pair_id'],t['label']))
            if d['status']!='accepted_with_recorded_limits' or not d['reason'] or d['crop_sha256']!=prior.file_sha256(Path(t['crop'])) or d['image_sha256']!=prior.file_sha256(Path(f['image'])):raise ValueError('Review blocked or stale')

def main():
    ep=OUT/'light-review/evidence.json';e=prior.read(ep);prior.verify(e);decisions=[]
    for f in e['events']:
        for t in f['targets']:
            key=(f['pair_id'],t['label'])
            decisions.append(dict(pair_id=key[0],label=key[1],reason=NOTES[key],status='accepted_with_recorded_limits',
                review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
                crop_sha256=prior.file_sha256(Path(t['crop'])),image_sha256=prior.file_sha256(Path(f['image']))))
    validate(e,decisions)
    prior.frozen(OUT/'light-review/review.json',dict(status='four_lighting_frames_explicitly_reviewed',decisions=decisions,
        training_ready=False,inputs={str(x):prior.file_sha256(x) for x in [ep,Path(__file__).resolve()]}))

if __name__=='__main__':main()
